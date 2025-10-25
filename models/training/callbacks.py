"""
Training progress callbacks using tqdm.
Works with HuggingFace Trainer to show live loss/step updates.
"""
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from tqdm.auto import tqdm

class TqdmLogger(TrainerCallback):
    """
    Minimal tqdm logger that:
      - Creates a progress bar over total training steps
      - Updates on 'log' events (which include 'loss', 'learning_rate', etc.)
      - Closes at the end of training
    """
    def __init__(self):
        self.pbar = None
        self._last_step = -1

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        total = state.max_steps if state.max_steps is not None else 0
        # If total is unknown, tqdm still works in indefinite mode
        self.pbar = tqdm(total=total, desc="training", dynamic_ncols=True, leave=True)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if self.pbar is None or logs is None:
            return
        # Update description with latest loss/learning rate if present
        loss = logs.get("loss", None)
        lr = logs.get("learning_rate", None)
        if loss is not None and lr is not None:
            self.pbar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{lr:.2e}"})
        elif loss is not None:
            self.pbar.set_postfix({"loss": f"{loss:.4f}"})
        # Move bar if we've advanced steps
        if state.global_step is not None and state.global_step > self._last_step:
            # Advance by the delta since last update
            delta = state.global_step - self._last_step
            self.pbar.update(delta)
            self._last_step = state.global_step

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.pbar is not None:
            self.pbar.set_description(f"training (epoch {state.epoch:.2f})")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
