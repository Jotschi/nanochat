Add native vision capabilities to nanochat while preserving its minimal, educational design philosophy.

Extend the nanochat repository with image understanding capabilities using a LLaMA-style multimodal architecture, while keeping the implementation faithful to nanochat’s core principles: minimal code, near-zero ML-library dependencies, transparent implementations, and components that can be trained and understood from first principles.

The agent should implement a vision-language model in which a lightweight vision encoder converts images into token-like embeddings that can be consumed by the existing LLaMA-style language model. Prefer implementing the required vision components directly in PyTorch rather than introducing large multimodal frameworks or dependencies. Reuse nanochat’s existing tokenizer, transformer blocks, training infrastructure, checkpointing, and inference interfaces wherever practical.

The resulting system should support:

Loading an image and combining it with a text prompt.
Encoding the image into a sequence of visual tokens compatible with the LLaMA transformer.
Autoregressive generation conditioned on both visual and textual tokens.
Training or fine-tuning the vision-language model with a simple, reproducible pipeline.
A minimal inference/demo interface showing image-question answering or image description.
Saving and loading multimodal checkpoints using nanochat’s existing conventions.

Architectural direction: use a small Vision Transformer (ViT-style) or similarly simple patch-based image encoder, project its output into the LLaMA hidden dimension, and feed the resulting visual embeddings into the language model as a contiguous visual-token prefix. Keep the language model architecture LLaMA-like (RMSNorm, RoPE, SwiGLU, causal self-attention), and avoid introducing a heavyweight multimodal architecture unless it is demonstrably necessary.

Dependency constraint: the implementation should aim for near-zero additional ML dependencies beyond those already used by nanochat. In particular, avoid Hugging Face Transformers, timm, torchvision-heavy model abstractions, or external VLM frameworks when equivalent functionality can be implemented in a small amount of native PyTorch code.

Engineering constraint: favor simplicity over state-of-the-art performance. The objective is not to build the strongest VLM, but to demonstrate that nanochat can be extended from text-only to vision-language modeling while retaining its “small enough to understand and train yourself” character.

Definition of done: a contributor should be able to clone nanochat, install essentially the existing dependencies, train or load a small multimodal checkpoint, provide an image plus a textual prompt, and obtain a generated textual response—with the entire vision-to-LLaMA path understandable by reading the repository rather than relying on an external multimodal framework.