import torch
import pytest
from model import DigitCNN


@pytest.fixture
def model():
    return DigitCNN()


def test_output_shape(model):
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)


def test_single_sample(model):
    x = torch.randn(1, 1, 28, 28)
    out = model(x)
    assert out.shape == (1, 10)


def test_output_not_all_equal(model):
    x = torch.randn(1, 1, 28, 28)
    out = model(x)
    assert not torch.all(out == out[0, 0]), "All logits are identical"


def test_different_inputs_different_outputs(model):
    x1 = torch.randn(1, 1, 28, 28)
    x2 = torch.randn(1, 1, 28, 28)
    out1 = model(x1)
    out2 = model(x2)
    assert not torch.allclose(out1, out2)


def test_gradient_flows(model):
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_eval_mode_deterministic(model):
    model.eval()
    x = torch.randn(2, 1, 28, 28)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2), "Eval mode should be deterministic"


def test_parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    assert total > 0
    # Sanity check: should be in a reasonable range for this architecture
    assert total < 10_000_000, "Model unexpectedly large"


def test_features_output_shape(model):
    x = torch.randn(2, 1, 28, 28)
    feat = model.features(x)
    assert feat.shape == (2, 128, 7, 7)
