What is **popeye**?
===================

**popeye** is an **open-source** project for developing, estimating, and validating pRF models. **popeye** was
created by `Kevin DeSimone <http://www.github.com/kdesimone>`_ with significant guidance and contributions from
`Ariel Rokem <http://www.github.com/arokem>`_. **popeye** was developed because there was no open-source software
for pRF modeling and estimation. The are several freely available MATLAB toolboxes for doing pRF model estimation,
including `mrVista <https://github.com/vistalab/vistasoft>`_ and `analyzePRF
<http://kendrickkay.net/analyzePRF/>`_, but since these toolboxes are written in MATLAB they are not truly
open-source. This motivated us to create **popeye** as project open to critique and colloboration.

The pRF model is what is known as a forward encoding model. We present the participant with a stimulus that varies
over time in the tuning dimension(s) of interest. If we're interested in the retinotopic organization of some
brain region, we would use a visual stimulus that varies over the horizontal and vertical dimensions of the visual
field. Often experimenters will use a sweeping bar stimulus that makes several passes through the visual field to
encode the visuotopic location of the measured response.

The **StimulusModel**, **PopulationModel**, and **PopulationFit** represent the fundamental units of **popeye**.
Each of these are defined in **popeye.base**. **StimulusModel** is an abstract class containing the stimulus
array—a NumPy ndarray—along with some supporting information about the stimulus timing and encoding.
**PopulationModel** takes a **StimulusModel** as input and defines a **generate_prediction** method for operating
on the stimulus via some user-defined receptive field model to produce a model prediction given some set of input
parameters. **PopulationFit** takes a **PopulationModel** and a data timeseries (1D NumPy ndarray), estimating the
the set of parameters of the **PopulationModel** that best fits the data.

