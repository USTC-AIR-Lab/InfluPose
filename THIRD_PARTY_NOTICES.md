# Third-party components

| Component | Use | License |
| --- | --- | --- |
| GenPose | NOCS data loading and augmentation conventions | [MIT](third_party/licenses/GenPose.txt) |
| PointNet++ CUDA operators | Sampling, grouping, gathering, and ball query | [MIT](third_party/pointnet2/LICENSE) |

The source in `third_party/pointnet2/src` and its Python interface are retained
with their upstream license. The build file packages the same extension as an
installable local dependency. No compiled binaries are distributed.

The data and benchmark conventions originate from NOCS and SPD preprocessing. Obtain datasets under their respective terms; the software
license does not grant rights to redistribute those datasets.
