"""
Anagrama Guider — prueba mínima de "Visual Anagrams" (Geng, Park, Owens 2024)
para ComfyUI. Dos prompts, dos vistas (identidad + una transformación).

En cada paso:  denoised = ( D(x, A)  +  V⁻¹( D(V(x), B) ) ) / 2
Como x0 = x − σ·v es lineal y V es una permutación (ortogonal), promediar las
predicciones x0 equivale exactamente a promediar velocidades/ruido.
Las vistas se aplican sobre el latente (últimas dos dims = alto, ancho),
así que vale para latentes 4D [B,C,H,W] y 5D [B,C,T,H,W] (VAE Qwen-Image).
"""

import torch
import comfy.samplers

H, W = -2, -1

VIEWS = {
    # nombre: (vista, inversa)
    "flip_vertical":   (lambda x: torch.flip(x, dims=(H,)),               lambda x: torch.flip(x, dims=(H,))),
    "flip_horizontal": (lambda x: torch.flip(x, dims=(W,)),               lambda x: torch.flip(x, dims=(W,))),
    "rotate_180":      (lambda x: torch.rot90(x, k=2, dims=(H, W)),       lambda x: torch.rot90(x, k=-2, dims=(H, W))),
    "rotate_90_cw":    (lambda x: torch.rot90(x, k=-1, dims=(H, W)),      lambda x: torch.rot90(x, k=1, dims=(H, W))),
}


class Guider_Anagrama(comfy.samplers.CFGGuider):
    def set_view(self, view_name):
        self.view_name = view_name
        self.view, self.inv = VIEWS[view_name]

    def set_conds(self, positive_a, positive_b, negative):
        # "positive" se mantiene como clave porque process_conds la usa de referencia
        self.inner_set_conds({"positive": positive_a, "positive_b": positive_b, "negative": negative})

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        if self.view_name == "rotate_90_cw" and x.shape[H] != x.shape[W]:
            raise ValueError(f"rotate_90_cw necesita latente cuadrado; recibido {x.shape[H]}x{x.shape[W]}")

        neg = self.conds.get("negative", None)
        out_a = comfy.samplers.sampling_function(
            self.inner_model, x, timestep, neg, self.conds.get("positive", None),
            self.cfg, model_options=model_options, seed=seed)
        out_b = comfy.samplers.sampling_function(
            self.inner_model, self.view(x), timestep, neg, self.conds.get("positive_b", None),
            self.cfg, model_options=model_options, seed=seed)
        return (out_a + self.inv(out_b)) * 0.5


class AnagramaGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive_a": ("CONDITIONING", {"tooltip": "Prompt de la imagen tal cual (identidad)"}),
                "positive_b": ("CONDITIONING", {"tooltip": "Prompt de la imagen transformada"}),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "view": (list(VIEWS.keys()), {"default": "flip_vertical"}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "B_RR_K_/anagrama"

    def get_guider(self, model, positive_a, positive_b, negative, cfg, view):
        guider = Guider_Anagrama(model)
        guider.set_conds(positive_a, positive_b, negative)
        guider.set_cfg(cfg)
        guider.set_view(view)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "AnagramaGuider": AnagramaGuider,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnagramaGuider": "Anagrama Guider (Visual Anagrams)",
}


# ---------------------------------------------------------------------------
# Variante: balanceo de vistas (idea de Anagram-MTL, sin CLIP)
# En vez de 50/50:  denoised = (1 − w_b)·D_A + w_b·V⁻¹(D_B)
#   w_b = clamp( weight_b + early_bias·σ + auto·(w_auto − 0.5), 0.05, 0.95 )
# early_bias: más peso a B al principio (cuando se decide la estructura); se
#   desvanece con σ. auto: w_auto = Δ_B / (Δ_A + Δ_B), con Δ = distancia entre
#   la x0 que pide cada vista y la x0 fusionada del paso anterior: la vista que
#   más quiere cambiar la imagen va perdiendo y recibe más peso (heurística
#   propia, no del paper). Sin pasadas extra.
# Con weight_b 0.5, early_bias 0 y auto 0 es idéntico al nodo base.
# ---------------------------------------------------------------------------
class Guider_AnagramaBalance(Guider_Anagrama):
    def set_balance(self, weight_b, early_bias, auto):
        self.weight_b, self.early_bias, self.auto = weight_b, early_bias, auto
        self.prev, self.prev_sigma = None, None

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        if self.view_name == "rotate_90_cw" and x.shape[H] != x.shape[W]:
            raise ValueError(f"rotate_90_cw necesita latente cuadrado; recibido {x.shape[H]}x{x.shape[W]}")

        neg = self.conds.get("negative", None)
        d_a = comfy.samplers.sampling_function(
            self.inner_model, x, timestep, neg, self.conds.get("positive", None),
            self.cfg, model_options=model_options, seed=seed)
        d_b = self.inv(comfy.samplers.sampling_function(
            self.inner_model, self.view(x), timestep, neg, self.conds.get("positive_b", None),
            self.cfg, model_options=model_options, seed=seed))

        s = timestep.reshape([-1] + [1] * (x.ndim - 1)).to(x.dtype)
        if self.prev_sigma is not None and float(s.max()) >= self.prev_sigma:
            self.prev = None                                         # σ sube: empieza un muestreo nuevo
        self.prev_sigma = float(s.max())

        w = self.weight_b + self.early_bias * s
        if self.auto > 0 and self.prev is not None and self.prev.shape == d_a.shape:
            da = (d_a - self.prev).flatten(1).norm(dim=1).reshape(s.shape)
            db = (d_b - self.prev).flatten(1).norm(dim=1).reshape(s.shape)
            w = w + self.auto * (db / (da + db).clamp(min=1e-8) - 0.5)
        w = w.clamp(0.05, 0.95)
        print(f"[AnagramaBalance] sigma={float(s.flatten()[0]):.3f}  w_b={float(w.flatten()[0]):.3f}")

        out = d_a * (1 - w) + d_b * w
        self.prev = out.detach()
        return out


class AnagramaGuiderBalance(AnagramaGuider):
    @classmethod
    def INPUT_TYPES(cls):
        t = AnagramaGuider.INPUT_TYPES()
        t["required"]["weight_b"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                               "tooltip": "Peso base de la vista B (0.5 = nodo base). Súbelo si B no aparece"})
        t["required"]["early_bias"] = ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05,
                                                 "tooltip": "Peso extra para B al principio, se desvanece con sigma. Negativo = favorece A"})
        t["required"]["auto"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                           "tooltip": "Balanceo automático: da más peso a la vista que va perdiendo. 0 = apagado"})
        return t

    def get_guider(self, model, positive_a, positive_b, negative, cfg, view, weight_b, early_bias, auto):
        guider = Guider_AnagramaBalance(model)
        guider.set_conds(positive_a, positive_b, negative)
        guider.set_cfg(cfg)
        guider.set_view(view)
        guider.set_balance(weight_b, early_bias, auto)
        return (guider,)


NODE_CLASS_MAPPINGS["AnagramaGuiderBalance"] = AnagramaGuiderBalance
NODE_DISPLAY_NAME_MAPPINGS["AnagramaGuiderBalance"] = "Anagrama Guider · Balance"
