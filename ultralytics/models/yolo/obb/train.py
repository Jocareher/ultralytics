# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG, RANK


class OBBTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.obb import OBBTrainer

        args = dict(model="yolov8n-obb.pt", data="dota8.yaml", epochs=3)
        trainer = OBBTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a OBBTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return OBBModel initialized with specified config and weights."""
        model = OBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of OBBValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "rot_loss", "vtx_loss"
        return yolo.obb.OBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args)
        )

    # --- New functions for OBB loss logging and progress display ---

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        If loss_items is None, returns a list of loss header names; if loss_items is provided,
        returns a dictionary mapping each loss name to its value.

        For the "obb" task, the expected losses are:
            [box_loss, cls_loss, dfl_loss, rot_loss, vtx_loss]
        """
        if loss_items is None:
            # This list is used to build the CSV header and plot labels.
            return [
                f"{prefix}/box_loss",
                f"{prefix}/cls_loss",
                f"{prefix}/dfl_loss",
                f"{prefix}/rot_loss",
                f"{prefix}/vtx_loss",
            ]
        else:
            # When loss_items is provided, return a dict for logging/saving.
            return {
                f"{prefix}/box_loss": loss_items[0].item(),
                f"{prefix}/cls_loss": loss_items[1].item(),
                f"{prefix}/dfl_loss": loss_items[2].item(),
                f"{prefix}/rot_loss": loss_items[3].item(),
                f"{prefix}/vtx_loss": loss_items[4].item(),
            }
