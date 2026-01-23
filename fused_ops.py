from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        # raise NotImplementedError
        A = input_values[0]
        B = input_values[1]
        max_res = torch.matmul(A, B)
        mean = max_res.mean(dim=-1, keepdim=True)
        var = max_res.var(dim=-1, keepdim=True, unbiased=False)
        normalized_res = (max_res - mean) / torch.sqrt(var + node.eps)
        return normalized_res

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        """TODO: your code here"""
        matmul_tes = matmul(node.inputs[0], node.inputs[1])
        x = matmul_tes
        dims = tuple(range(-len(node.normalized_shape), 0))
        x_mean = mean(x, dim=dims, keepdim=True)
        x_mu = x - x_mean
        x_mu_pow = power(x_mu, 2)
        x_var = mean(x_mu_pow, dim=dims, keepdim=True) + node.eps

        dxhat = output_grad
        term1 = mean(dxhat, dim=dims, keepdim=True)
        term2 = mean(dxhat * (x_mu), dim=dims, keepdim=True)
        std_inv = power(sqrt(x_var), -1.0)

        grad_x = std_inv * (output_grad - term1 - x_mu * (term2 / x_var))

        grad_A = matmul(grad_x, transpose(node.inputs[1], -2, -1))
        grad_B = matmul(transpose(node.inputs[0], -2, -1), grad_x)

        return [grad_A, grad_B]

        



class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        A = input_values[0]
        B = input_values[1]
        max_res = torch.matmul(A, B)
        max_v = torch.max(max_res, dim=node.dim, keepdim=True)[0]
        exps = torch.exp(max_res - max_v)
        return exps / torch.sum(exps, dim=node.dim, keepdim=True)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        """TODO: your code here"""
        matmul_tes = matmul(node.inputs[0], node.inputs[1])
        input_node = matmul_tes
        softmax_output = softmax(input_node)
        grad_input = softmax_output * (output_grad - sum_op(output_grad * softmax_output, dim=node.dim, keepdim=True))

        grad_A = matmul(grad_input, transpose(node.inputs[1], -2, -1))
        grad_B = matmul(transpose(node.inputs[0], -2, -1), grad_input)

        return [grad_A, grad_B]
        

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()