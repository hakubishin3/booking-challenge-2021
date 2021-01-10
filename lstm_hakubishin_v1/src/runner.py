import torch
from catalyst import metrics
from catalyst.dl import Runner
from catalyst.dl.utils import any2device


class CustomRunner(Runner):
    def _handle_batch(self, batch):
        (
            city_id_tensor,
            booker_country_tensor,
            device_class_tensor,
            affiliate_id_tensor,
            y,
        ) = batch
        out = self.model(
            city_id_tensor,
            booker_country_tensor,
            device_class_tensor,
            affiliate_id_tensor,
        )
        loss = self.criterion(out, y.view(y.size(0) * y.size(1)))
        accuracy01, accuracy04 = metrics.accuracy(
            out, y.view(y.size(0) * y.size(1)), topk=(1, 4)
        )
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
        if len(batch) == 4:
            (
                city_id_tensor,
                booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
            ) = batch
        elif len(batch) == 5:
            (
                city_id_tensor,
                booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
                y,
            ) = batch
        else:
            raise RuntimeError
        out = self.model(
            city_id_tensor,
            booker_country_tensor,
            device_class_tensor,
            affiliate_id_tensor,
        )
        return out
