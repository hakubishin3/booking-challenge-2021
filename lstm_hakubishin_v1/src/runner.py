import torch
from catalyst import metrics
from catalyst.dl import Runner
from catalyst.dl.utils import any2device


class CustomRunner(Runner):
    def _handle_batch(self, batch):
        input_tensors = batch[:-1]
        y = batch[-1]
        out = self.model(*input_tensors)
        loss = self.criterion(out, y)
        accuracy01, accuracy04 = metrics.accuracy(out, y, topk=(1, 4))
        self.batch_metrics.update(
            {"loss": loss, "accuracy01": accuracy01, "accuracy04": accuracy04}
        )
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    @torch.no_grad()
    def predict_batch(self, batch):
        batch = any2device(batch, self.device)
        if len(batch) == 15:
            input_tensors = batch
        elif len(batch) == 16:
            input_tensors = batch[:-1]
        else:
            raise RuntimeError

        out = self.model(*input_tensors)
        return out
