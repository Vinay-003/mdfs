
import torch
from PIL import Image

from model import EffNet, APL, prepare_image


def cov(m, rowvar=False):
    """
    Exact covariance implementation used by upstream test.py.
    """
    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")

    if m.dim() < 2:
        m = m.view(1, -1)

    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1)

    m = m - torch.mean(
        m,
        dim=1,
        keepdim=True
    )

    mt = m.t()

    return fact * m.matmul(mt).squeeze()


class MDFSScorer:

    def __init__(
        self,
        weights_path,
        device="cuda"
    ):

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.device = torch.device(device)

        print("MDFS device:", self.device)

        if self.device.type == "cuda":
            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )

        print("Loading EfficientNet-B7...")

        self.vgg = EffNet().to(
            self.device
        ).eval()

        self.apl = APL().to(
            self.device
        ).eval()

        print("Loading reference features...")

        reference = torch.load(
            weights_path,
            map_location=self.device,
            weights_only=False
        )

        print(
            "Reference tensor:",
            tuple(reference.shape)
        )

        print(
            "Computing reference mean..."
        )

        self.reference_mean = reference.mean(
            0,
            keepdim=True
        )

        print(
            "Computing reference covariance..."
        )

        self.reference_cov = cov(reference)

        # Raw ~613MB reference tensor is no longer needed.
        del reference

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        print("MDFS scorer ready.")


    @torch.inference_mode()
    def score_pil(self, image):

        image = image.convert("RGB")

        x = prepare_image(
            image
        ).to(self.device)

        features = self.vgg(x)

        out, ps = self.apl(
            features,
            select=0
        )

        # Exact upstream mvg() calculation
        x_cov = cov(out)

        delta = (
            out -
            self.reference_mean
        )

        solution = torch.linalg.solve(
            (
                x_cov +
                self.reference_cov
            ) / 2,
            delta.t()
        )

        distances = delta.mm(
            solution
        )

        distances = torch.diagonal(
            distances
        ).abs().sqrt()

        score_map = distances.reshape(
            ps.shape
        )

        score = (
            score_map * ps
        ).sum() / ps.sum()

        return float(score.item())


    def score_path(self, path):

        with Image.open(path) as image:
            return self.score_pil(image)
