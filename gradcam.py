import numpy as np
import cv2

class GradCAM_ResNetUNet:
    def __init__(self, model):
        self.model = model
        self.model.eval()

        self.target_layer = model.up1.conv.conv[0]

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def _register_hooks(self):
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def __call__(self, img_tensor, class_id=None):

        logits = self.model(img_tensor)

        if class_id is None:
            class_id = logits.argmax(1)

        loss = logits[0, class_id[0]].mean()

        self.model.zero_grad()
        loss.backward()

        acts = self.activations[0]
        grads = self.gradients[0]

        weights = grads.mean(dim=(1, 2))

        cam = (weights[:, None, None] * acts).sum(dim=0)

        cam = cam.cpu().numpy()
        cam = np.nan_to_num(cam)

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        H, W = img_tensor.shape[2], img_tensor.shape[3]
        cam = cv2.resize(cam, (W, H))

        return cam