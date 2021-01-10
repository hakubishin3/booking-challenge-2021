import torch
from catalyst import metrics
from catalyst.dl import Runner
from catalyst.dl.utils import any2device


class CustomRunner(Runner):
    def _handle_batch(self, batch):
        city_id_tensor, booker_country_tensor, y = batch
        out, hidden = self.model(city_id_tensor, booker_country_tensor)
        loss = self.criterion(out, y.view(y.size(0) * y.size(1)))
        accuracy01, accuracy04 = metrics.accuracy(
            out, y.view(y.size(0) * y.size(1)), topk=(1, 4)
        )
        self.batch_metrics.update(
            {"loss": loss, "accuracy01": accuracy01, "accuracy04": accuracy04}
        )
        if self.is_train_loader:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()
            self.optimizer.zero_grad()

    @torch.no_grad()
    def predict_batch(self, batch):
        batch = any2device(batch, self.device)
        if len(batch) == 2:
            city_id_tensor, booker_country_tensor = batch
        elif len(batch) == 3:
            city_id_tensor, booker_country_tensor, y = batch
        else:
            raise RuntimeError
        out, hidden = self.model(city_id_tensor, booker_country_tensor)
        return out
