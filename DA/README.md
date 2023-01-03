The code to load the data is taken from https://github.com/MultiScale-BCI/IV-2a.

Instruction: Download the dataset "Four class motor imagery (001-2014)" from [BCI Competion IV-2a](http://bnci-horizon-2020.eu/database/data-sets). Put them in a folder dataset at the root, or change the path.

List of notebooks:
- Baseline: Baseline obtained with the method of https://github.com/MultiScale-BCI/IV-2a
- Baseline_Classifier: Training on source + accuracy on target without adaptation on all subjects (TODO: add batchnorm to classifier?)
- Baseline_Classifier_Subect1: Same on Subject 1.
- BCI_Subject1 and Test_Subject1: Domain adaptation (old version, with SPDNet from [Brooks, Daniel, et al. "Riemannian batch normalization for SPD neural networks." Advances in Neural Information Processing Systems 32 (2019).](https://proceedings.neurips.cc/paper/2019/hash/6e69ebbfad976d4637bb4b39de261bf7-Abstract.html), problem with training)
- Basic_Transformation: only use translation and rotations (on the source).
- Cross_subject: Test of cross subject.
- In folder multifreq: Code where we use multifrequences with a kernel sum.
