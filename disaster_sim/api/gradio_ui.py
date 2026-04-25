from __future__ import annotations

from typing import Any, Callable, Optional

import gradio as gr


UIResult = dict[str, Any]
ResetFn = Callable[[Optional[int], Optional[str]], UIResult]
StepFn = Callable[[str, int, int], UIResult]
StateFn = Callable[[], dict[str, Any]]


def create_gradio_demo(
    reset_fn: ResetFn,
    step_fn: StepFn,
    state_fn: StateFn,
    grid_size: int,
) -> gr.Blocks:
    """Build a Gradio panel that controls reset and step operations."""

    def _format_outputs(payload: UIResult):
        observation = payload.get("observation", {})
        masked_telemetry = observation.get("masked_telemetry", [])

        return (
            float(payload.get("reward", 0.0)),
            bool(payload.get("done", False)),
            int(observation.get("visible_signals", 0)),
            payload.get("state", {}),
            observation,
            masked_telemetry,
        )

    def on_reset(seed_text: str, episode_id_text: str):
        seed_value: Optional[int] = None
        seed_text = (seed_text or "").strip()
        if seed_text:
            try:
                seed_value = int(seed_text)
            except ValueError as exc:
                raise gr.Error("Seed must be an integer value.") from exc

        episode_id = (episode_id_text or "").strip() or None
        payload = reset_fn(seed_value, episode_id)
        return _format_outputs(payload)

    def on_step(interaction: str, row: float, col: float):
        payload = step_fn(interaction, int(row), int(col))
        return _format_outputs(payload)

    with gr.Blocks(title="Disaster Simulation Control Panel") as demo:
        gr.Markdown(
            "# Disaster Simulation Control Panel\n"
            "Use reset to start an episode, then step through actions."
        )

        with gr.Row():
            seed_input = gr.Textbox(label="Seed (optional)", placeholder="e.g., 42")
            episode_id_input = gr.Textbox(
                label="Episode ID (optional)",
                placeholder="custom episode identifier",
            )
            reset_btn = gr.Button("Reset", variant="primary")

        with gr.Row():
            interaction_input = gr.Dropdown(
                choices=["dispatch", "suppress", "wait"],
                value="wait",
                label="Interaction",
            )
            row_input = gr.Slider(
                minimum=0,
                maximum=max(0, grid_size - 1),
                value=0,
                step=1,
                label="Row",
            )
            col_input = gr.Slider(
                minimum=0,
                maximum=max(0, grid_size - 1),
                value=0,
                step=1,
                label="Col",
            )
            step_btn = gr.Button("Step", variant="secondary")

        with gr.Row():
            reward_output = gr.Number(label="Reward", value=0.0)
            done_output = gr.Checkbox(label="Done", value=False)
            visible_signals_output = gr.Number(label="Visible Signals", value=0)

        state_output = gr.JSON(label="Environment State")
        observation_output = gr.JSON(label="Observation Payload")
        telemetry_grid_output = gr.Dataframe(
            headers=[str(idx) for idx in range(grid_size)],
            datatype="number",
            row_count=(grid_size, "fixed"),
            column_count=(grid_size, "fixed"),
            interactive=False,
            label="Masked Telemetry Grid",
        )

        refresh_state_btn = gr.Button("Refresh State")

        reset_btn.click(
            on_reset,
            inputs=[seed_input, episode_id_input],
            outputs=[
                reward_output,
                done_output,
                visible_signals_output,
                state_output,
                observation_output,
                telemetry_grid_output,
            ],
        )

        step_btn.click(
            on_step,
            inputs=[interaction_input, row_input, col_input],
            outputs=[
                reward_output,
                done_output,
                visible_signals_output,
                state_output,
                observation_output,
                telemetry_grid_output,
            ],
        )

        refresh_state_btn.click(fn=state_fn, inputs=None, outputs=state_output)
        demo.load(fn=state_fn, inputs=None, outputs=state_output)

    return demo
