"""
Aquarelab: recolor any photo with a new palette.

Run from the project root:

    uv run --group app python app/app.py

The model path can be changed with the AQUARELAB_MODEL environment variable.
"""

import os

import gradio as gr
import numpy as np

from aquarelab.inference import (
    DEFAULT_THRESHOLD,
    analyze,
    load_model,
    recolor_tones,
    tones_in_use,
)
from aquarelab.recolor import (
    assign_by_luminance,
    hex_to_lab,
    lab_to_hex,
    normalize_hex,
    parse_hex_colors,
)

MODEL_PATH = os.environ.get("AQUARELAB_MODEL", "models/palettenet_v1.safetensors")
MAX_TONES = 8

# Each preset stays within one color family. Palettes that mix opposite hues
# can split smooth gradients into visible bands, so they are left to custom input.
PRESETS = {
    "Ocean": ["#03045e", "#023e8a", "#0077b6", "#0096c7", "#00b4d8", "#48cae4", "#90e0ef", "#caf0f8"],
    "Forest": ["#081c15", "#1b4332", "#2d6a4f", "#40916c", "#52b788", "#74c69d", "#95d5b2", "#d8f3dc"],
    "Autumn": ["#3d1f0d", "#6b2d0f", "#9c3d10", "#c8551a", "#e07a2f", "#eda44c", "#f4c77a", "#faebc8"],
    "Pastel": ["#5e548e", "#9f86c0", "#be95c4", "#e0b1cb", "#f4c2c2", "#fcd5ce", "#fde2e4", "#fff1e6"],
}

THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="pink",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

CSS = """
.gradio-container { max-width: 100% !important; padding: 12px 20px !important; }
footer { display: none !important; }
#layout { gap: 20px; align-items: stretch; }
#panel { gap: 12px; }
#header h1 { margin: 0; font-size: 1.7rem; line-height: 1.2; }
#header p { margin: 4px 0 0; opacity: 0.75; }
.section-title { font-weight: 600; font-size: 0.95rem; margin: 4px 0 0; line-height: 1.4; }
.section-title span { font-weight: 400; opacity: 0.7; }
#tones, #presets { display: grid !important; grid-template-columns: repeat(4, 1fr); gap: 8px; }
#tones > *, #presets > * { min-width: 0 !important; }
#photo img { object-fit: scale-down !important; }
"""

model = load_model(MODEL_PATH, device="cpu")


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0, 1) * 255).round().astype(np.uint8)


def hidden_pickers() -> list:
    return [gr.ColorPicker(visible=False) for _ in range(MAX_TONES)]


def picker_updates(state: dict, colors: list[str]) -> list:
    """One swatch per tone in use, darkest first, labeled with the extracted hex value."""
    updates = []
    for i in range(MAX_TONES):
        if i < len(state["order"]):
            updates.append(
                gr.ColorPicker(value=colors[i], label=state["original"][i], visible=True)
            )
        else:
            updates.append(gr.ColorPicker(visible=False))
    return updates


def on_upload(image):
    if image is None:
        return [None, None, "", *hidden_pickers()]
    analysis = analyze(model, image)
    order = tones_in_use(analysis)
    original = [lab_to_hex(analysis.palette[t]) for t in order]
    state = {"analysis": analysis, "order": order, "original": original}
    return [state, to_uint8(analysis.image), " ".join(original), *picker_updates(state, original)]


def on_clear():
    return [None, "", *hidden_pickers()]


def on_picker(state, *colors):
    """Recolor only the tones whose swatch differs from the extracted color."""
    if state is None:
        return [gr.skip(), gr.skip()]
    current = [normalize_hex(c) for c in colors[: len(state["order"])]]
    replacements = {
        tone: color
        for tone, original, color in zip(state["order"], state["original"], current)
        if color != original
    }
    return [to_uint8(recolor_tones(state["analysis"], replacements)), " ".join(current)]


def apply_colors(colors: list[str], state):
    """Match a whole palette to the tones by luminance and show it in the swatches."""
    if state is None:
        raise gr.Error("Upload a photo first.")
    order = state["order"]
    if len(colors) < len(order):
        raise gr.Error(f"This photo has {len(order)} tones. Provide at least {len(order)} colors.")
    analysis = state["analysis"]
    matched = assign_by_luminance(
        analysis.palette, hex_to_lab(colors), usage=analysis.owned, min_usage=DEFAULT_THRESHOLD
    )
    new_colors = [lab_to_hex(matched[t]) for t in order]
    image = recolor_tones(analysis, dict(zip(order, new_colors)))
    return [to_uint8(image), " ".join(new_colors), *picker_updates(state, new_colors)]


def on_custom_palette(state, text):
    return apply_colors(parse_hex_colors(text), state)


def on_reset(state):
    if state is None:
        return [gr.skip(), "", *hidden_pickers()]
    original = state["original"]
    return [to_uint8(state["analysis"].image), " ".join(original), *picker_updates(state, original)]


def section(title: str, hint: str = "") -> gr.HTML:
    """Short section title. HTML avoids the scroll arrows and clipping of Markdown blocks."""
    hint_html = f" <span>· {hint}</span>" if hint else ""
    return gr.HTML(f'<div class="section-title">{title}{hint_html}</div>')


with gr.Blocks(title="Aquarelab", theme=THEME, css=CSS, fill_height=True) as demo:
    state = gr.State(None)

    with gr.Row(elem_id="layout", equal_height=False):
        with gr.Column(scale=3):
            photo = gr.Image(
                type="pil",
                sources=["upload"],
                format="png",
                show_label=False,
                height="calc(100vh - 48px)",
                elem_id="photo",
            )

        with gr.Column(scale=1, min_width=380, elem_id="panel"):
            gr.HTML(
                '<div id="header"><h1>Aquarelab</h1>'
                "<p>Recolor any photo. Light, shadow and texture are kept.</p></div>"
            )

            section("Extracted tones", "click a color to edit it")
            with gr.Row(elem_id="tones"):
                pickers = [
                    gr.ColorPicker(visible=False, interactive=True, min_width=0)
                    for _ in range(MAX_TONES)
                ]

            current_palette = gr.Textbox(
                label="Current palette", interactive=False, lines=2, max_lines=2
            )

            section("Change everything at once")
            with gr.Row(elem_id="presets"):
                preset_buttons = {name: gr.Button(name, size="sm", min_width=0) for name in PRESETS}
            with gr.Row():
                palette_text = gr.Textbox(
                    show_label=False,
                    placeholder="Paste a palette: #1b4332 #40916c #95d5b2 ...",
                    scale=4,
                )
                apply_button = gr.Button("Apply", variant="primary", scale=1, min_width=0)

            reset_button = gr.Button("Reset colors", variant="secondary", size="sm")

    pickers_out = [photo, current_palette, *pickers]

    photo.upload(on_upload, photo, [state, *pickers_out])
    photo.clear(on_clear, None, [state, current_palette, *pickers])
    for picker in pickers:
        picker.input(on_picker, [state, *pickers], [photo, current_palette])
    for name, button in preset_buttons.items():
        button.click(lambda s, colors=PRESETS[name]: apply_colors(colors, s), state, pickers_out)
    apply_button.click(on_custom_palette, [state, palette_text], pickers_out)
    palette_text.submit(on_custom_palette, [state, palette_text], pickers_out)
    reset_button.click(on_reset, state, pickers_out)


if __name__ == "__main__":
    demo.launch()