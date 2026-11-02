# DCVC-UF HSI Transfer and BPPPC Validation

## Protocol

- Source model: DCVC-UF HT-S, official CVPR 2026 checkpoint
- Dataset: HySpecNet-11k, official hard split
- Training subset: 512 patches from the hard train split
- Validation subset: 64 patches from the hard validation split
- Test set: all 91 patches in the hard test split
- Patch size: 128 x 128
- Input: 8-bit pseudo-YUV grayscale bands (`Y = value - 0.5`, `U = V = 0`)
- Evaluated bands: first 200 of 202 valid bands, forming 25 complete 8-band chunks
- Rate points: QP 10, 31, 52, and 63
- Bitstream: actual arithmetic-coded bytes, including NAL/SPS headers

For HSI, one band is one channel. Therefore:

```text
bpppc = total bitstream bits / (128 * 128 * 200)
```

This is also numerically bpppb when a band is treated as a channel.

## Results

All values are means over the 91 hard-test patches.

| QP | Baseline bpppc | Baseline PSNR | 20-epoch transfer bpppc | Transfer PSNR |
|---:|---:|---:|---:|---:|
| 10 | 0.00129743 | 29.4373 dB | 0.00080599 | 27.1026 dB |
| 31 | 0.00473413 | 33.5490 dB | 0.00160220 | 29.4676 dB |
| 52 | 0.01677511 | 37.8632 dB | 0.00810014 | 33.8236 dB |
| 63 | 0.03083860 | 40.1095 dB | 0.01974149 | 35.8353 dB |

## Conclusion

The 20-epoch transfer run reduces bpppc at every tested QP, but it also
reduces PSNR by roughly 2.3 to 4.3 dB. At comparable quality points, the
original video-pretrained DCVC-UF dominates the fine-tuned model in this
experiment.

This is therefore a valid metric and pipeline validation, but not evidence
that the current fine-tuning recipe improves HSI rate-distortion performance.
The most likely causes are the small training subset, the high official
from-scratch stage0 learning rate used for transfer, and the QP-dependent
quantization parameters shifting during adaptation.

## Artifacts

- Baseline test JSON: `runs/bpppc_baseline_91.json`
- Baseline QP63 JSON: `runs/bpppc_baseline_q63_91.json`
- Transfer model test JSON: `runs/bpppc_final_91.json`
- Transfer checkpoint: `runs/hyspecnet_hts_stage0/ckpt.pth.tar`
- Training log: `logs/hyspecnet_stage0_20.log`
- Test log: `logs/bpppc_final_91.log`

## Limitations

- This uses 512 of the 8,035 hard-train patches and 20 of the official
  45 stage0 epochs.
- The first 200 bands are evaluated; the final two valid bands are omitted
  to preserve complete 8-band chunks without padding.
- PSNR is reported, but SAM, ERGAS, and MRAE still need to be added for a
  complete spectral-fidelity evaluation.
- Published HySpecNet-11k comparisons mostly use the easy split, so their
  values are not directly comparable to this hard-split result.
