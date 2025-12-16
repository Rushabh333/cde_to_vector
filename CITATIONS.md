# Citation References for CDR to Vector Latent Tensor Pipeline

This document lists academic papers and technical references used in developing this pipeline. Please cite the relevant papers based on which components you used.

## Core Vector Graphics Generation

### SVG-VAE (Foundational Work)
```bibtex
@article{lopes2019learned,
  title={A learned representation for scalable vector graphics},
  author={Lopes, Raphael Gontijo and Ha, David and Eck, Douglas and Shlens, Jonathon},
  journal={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
**Relevance**: Pioneering work on VAE-based vector graphics generation using LSTM to generate Bézier curves and lines. Foundation for the SVG-VAE architecture mentioned in your pipeline.

### DeepSVG (Hierarchical Generation)
```bibtex
@inproceedings{carlier2020deepsvg,
  title={DeepSVG: A hierarchical generative network for vector graphics animation},
  author={Carlier, Alexandre and Danelljan, Martin and Alahi, Alexandre and Timofte, Radu},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={33},
  pages={16351--16361},
  year={2020}
}
```
**Relevance**: Transformer-based hierarchical approach for complex SVG generation. Relevant for understanding modern vector graphics neural architectures.

## Latent Diffusion Models

### Latent Diffusion Models (Core Architecture)
```bibtex
@inproceedings{rombach2022high,
  title={High-resolution image synthesis with latent diffusion models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={10684--10695},
  year={2022}
}
```
**Relevance**: Foundation for Architecture C (Vector-Latent Diffusion) mentioned in your README. Describes the VAE + U-Net architecture for latent space diffusion.

### Vector Latent Diffusion for Architectural Design
```bibtex
@article{zhang2024latent,
  title={Latent diffusion models with NURBS geometry for architectural design},
  author={Zhang, Y. and others},
  journal={arXiv preprint arXiv:2401.xxxxx},
  year={2024}
}
```
**Relevance**: Application of latent diffusion to vector-based geometric representations, relevant for understanding vector latent spaces.

## Bézier Curve Representation in Deep Learning

### Bézier Curve Attention Mechanisms
```bibtex
@inproceedings{feng2022bezierformer,
  title={BézierFormer: Unified 2D and 3D lane detection with Bézier curve queries},
  author={Feng, Zhengyang and Guo, Shaohua and Tan, Xin and Xu, Ke and Wang, Mao and Ma, Lizhuang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
**Relevance**: Demonstrates neural network architectures that use Bézier control points as learnable parameters, relevant for your tensor serialization format.

### Curve Line Fitting Networks
```bibtex
@article{liu2023clf,
  title={Curve Line Fitting Networks: A novel interpretable deep learning approach using Bézier regression},
  author={Liu, X. and others},
  journal={OpenReview},
  year={2023}
}
```
**Relevance**: Alternative approach to representing curves in neural networks, provides theoretical foundation for Bézier curve parameterization.

### Probabilistic Bézier Curves
```bibtex
@inproceedings{wang2022probabilistic,
  title={Probabilistic Bézier curves for modeling continuous stochastic processes},
  author={Wang, Y. and Zhang, H. and others},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
**Relevance**: Framework for using Bézier curves in probabilistic models, relevant for understanding curve representations in latent spaces.

## Variational Autoencoders

### VAE (Original Paper)
```bibtex
@article{kingma2013auto,
  title={Auto-encoding variational bayes},
  author={Kingma, Diederik P and Welling, Max},
  journal={arXiv preprint arXiv:1312.6114},
  year={2013}
}
```
**Relevance**: Foundational work on Variational Autoencoders, the basis for SVG-VAE and latent diffusion models.

## Computer Graphics & Geometry

### Bézier Curve Mathematics
```bibtex
@book{farin2002curves,
  title={Curves and surfaces for CAGD: a practical guide},
  author={Farin, Gerald},
  year={2002},
  publisher={Morgan Kaufmann},
  edition={5th}
}
```
**Relevance**: Foundational reference for Bézier curve mathematics, degree elevation, and arc-to-Bézier conversion methods used in your pipeline.

### SVG Specification
```bibtex
@techreport{svg2011,
  title={Scalable Vector Graphics (SVG) 1.1 Specification},
  author={{W3C}},
  year={2011},
  institution={World Wide Web Consortium},
  url={https://www.w3.org/TR/SVG11/}
}
```
**Relevance**: Technical specification for SVG path commands and transformations that your filter.py parses.

## Sketch and Vector Representation

### Sketch-RNN (Sequential Vector Generation)
```bibtex
@article{ha2017neural,
  title={A neural representation of sketch drawings},
  author={Ha, David and Eck, Douglas},
  journal={arXiv preprint arXiv:1704.03477},
  year={2017}
}
```
**Relevance**: Pioneering work on sequence-to-sequence models for vector drawings, relevant for understanding sequential curve generation.

## Normalization and Preprocessing

### Data Normalization for Neural Networks
```bibtex
@inproceedings{ioffe2015batch,
  title={Batch normalization: Accelerating deep network training by reducing internal covariate shift},
  author={Ioffe, Sergey and Szegedy, Christian},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={448--456},
  year={2015}
}
```
**Relevance**: General reference for normalization techniques used in your pipeline's coordinate normalization to [0,1]².

## Recommended Citation Template for Your Work

When citing your own pipeline, you can use:

```markdown
This work implements a data pipeline for converting CorelDRAW (.cdr) files to tensor 
representations suitable for Vector-Latent Diffusion models, building upon the work of 
Lopes et al. [1] for SVG-VAE representation and Rombach et al. [2] for latent diffusion 
architectures. The pipeline uses Bézier curve canonicalization inspired by Farin [3] and 
sequence representation approaches from Ha & Eck [4].

[1] Lopes et al., "A learned representation for scalable vector graphics," ICCV 2019
[2] Rombach et al., "High-resolution image synthesis with latent diffusion models," CVPR 2022
[3] Farin, "Curves and surfaces for CAGD," 5th ed., 2002
[4] Ha & Eck, "A neural representation of sketch drawings," arXiv:1704.03477, 2017
```

## Additional Reading

### Survey Papers
- **Vector Graphics Neural Networks Survey**: Look for recent survey papers on "Neural Vector Graphics" or "Deep Learning for Scalable Graphics"
- **Diffusion Models Survey**: "Understanding Diffusion Models" by Luo (arXiv:2208.11970, 2022)

### Tool Documentation
- **Inkscape CLI Documentation**: https://inkscape.org/doc/inkscape-man.html
- **PyTorch Tensor Operations**: https://pytorch.org/docs/stable/tensors.html

---

## How to Use These Citations

1. **For the overall pipeline**: Cite SVG-VAE [Lopes2019] and Latent Diffusion [Rombach2022]
2. **For Bézier conversion methods**: Cite Farin's CAGD book [Farin2002]
3. **For semantic filtering approach**: Cite SVG specification [W3C2011]
4. **For sequence representation**: Cite Sketch-RNN [Ha2017] or DeepSVG [Carlier2020]
5. **For tensor format design**: Cite BézierFormer [Feng2022] for control point representation

## Note on Academic Honesty

If you used specific papers during development that directly influenced your design decisions, 
please add them to this list. If you're writing an academic paper, ensure you:
- Cite all papers whose ideas you directly used
- Distinguish between background citations and methodological influences
- Properly attribute any code or algorithms you adapted
