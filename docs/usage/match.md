Waveform matches & friends
==========================

A search for a GW signal aims to determine whether a signal at the interferometer {math}`s` is composed by pure noise or by noise+signal.
A search statistics measures the probability of observing a signal against the probability of observing pure noise. 
It is a function of a _complex_ scalar product between two waveforms {math}`h_1, h_2`:

```{math}
  <h_1|h_2> = 4 \int \textrm{d}f \;\; \frac{\tilde{h}^*_1(f) \tilde{h}_2(f)}{S_n(f)}
```
where {math}`S_n(f)` is the power spectral density of the noise and {math}`\tilde{\phantom{h}}` indicates the Fourier transform. We denote by {math}`(\cdot|\cdot)` the real part of the complex scalar product and {math}`[\cdot|\cdot]` the imaginary part.

More formally, the search statistics is the probability ratio {math}`\frac{p(s|n+h)}{p(s|n)}` between the signal hypotesis and the noise hypothesis. The signal hyphothesis is the probability that a test waveform waveform, called _template_, with polarizations {math}`h_+, h_\times`, is present in the observed data {math}`s`. According to this hypothesis, the signal at the interferometer is a linear combination of the two WF polarization plus some noise {math}`n`: {math}`s = F_+ h_+ + F_\times h_\times + n`. 
The noise hyphothesis is the probability the observed data are only composed by noise {math}`n`: {math}`s = n`.
A search aims to find the maximum of the search statistics over a large parameter space. Depending on specific assumpions on the nature of the signal, the likelihood can be maximised over a number of nuisance variables:

```{math}
  \max \log\frac{p(s|n+h)}{p(s|n)}=\left\{
    \begin{array}{@{} l c @{}}
      (s|\hat{h}_+)^2 + (s|\hat{h}_\times)^2   & \text{non precessing} \\
      \frac{1}{1-(\hat{h}_+|\hat{h}_\times)^2} \left[(s|\hat{h}_+)^2 + (s|\hat{h}_\times)^2 - 2(s|\hat{h}_+)(s|\hat{h}_\times)(\hat{h}_+|\hat{h}_\times)  \right] & \text{precessing/HM}
    \end{array}\right.
```

The search statistics is computed using _normalized_ waveforms {math}`\hat{h}`, such that {math}`\hat{h} = \frac{h}{(h|h)}`. If a signal in the data exactly matches a template, the search statistics above is equivalent to the quantity {math}`<s|s>`, also called Signal to noise ratio (SNR).
The quantities above are _all_ function of a constant time shift between templates and signal, hence one needs to carry out the explicit maximisation over time (matched filtering). In all the expressions here, we assume that the time maximisation is understood and we leave it out from our notation.

The differences between precessing/HM and non-precessing case are due to the fact that, unlike the general case, in the non precessing case, the two polarizations {math}`h_+, h_\times` are related by a simple symmetry (in frequency domain) {math}`\tilde{h}_+ = i \tilde{h}_\times`: this yields to a simpler expression. So far, most of the searches have been using the simple version.

A _template bank_ aims to maximise by brute force the likelihood above by computing its value over a large number of templates. The goal of `mbank` is to generate and validate a template bank, making it directly usable for searches.

## Match(es) definition

Motivated by the maximised search likelihood above, one can define a _match_ between a signal {math}`s = F_+ h^\text{signal}_+ + F_\times h^\text{signal}_\times` and a template with polarization {math}`h_+, h_\times`.
The match takes values in {math}`(0,1]`.

For non-precessing signals it holds {math}`(\hat{s}|\hat{h}_\times) = [\hat{s}|\hat{h}_+]` and the match is defined as:

```{math}
\mathcal{M}_{\text{std}} = \sqrt{ (\hat{s}|\hat{h}_+)^2 + [\hat{s}|\hat{h}_+]^2 } 
```
and can be computed with `mbank` using function {func}`mbank.metric.cbc_metric.WF_match`.

For the general signal, the match is also called _symphony_ match, after the title of the [paper](https://arxiv.org/abs/1709.09181) that first introduced it. It is defined as:

```{math}
\mathcal{M}_{\text{symphony}} = \sqrt{ \frac{1}{1-(\hat{h}_+|\hat{h}_\times)^2} \left[(\hat{s}|\hat{h}_+)^2 + (\hat{s}|\hat{h}_\times)^2 - 2(\hat{s}|\hat{h}_+)(\hat{s}|\hat{h}_\times)(\hat{h}_+|\hat{h}_\times)  \right]  }
```
It can be computed with {func}`mbank.metric.cbc_metric.WF_symphony_match`.

The match can be seen as a function of the parameters {math}`\theta_1, \theta_2` of two signals, by considering, in the equations above {math}`s(\theta_1) = F_+ h_+(\theta_1) + F_\times h_\times(\theta_1)` and {math}`h_+(\theta_2), h_\times(\theta_2)`.
This is computed by function {func}`mbank.metric.cbc_metric.match`. Of course, in this case the match has a parametric dependence on the antenna patterns (or equivalently on the sky location of the GW source).


The match has a simple physical interpretation: it amounts to the fraction of SNR lost by filtering the signal {math}`s` with the template {math}`h_+, h_\times`.

The match can be used to validate the performance of the bank, by evaluating the match on a random set of points, called injections.

## A distance between templates

The definitions given above are not (yet) useful to define a distance between two points {math}`\theta_1, \theta_2` of the signal manifold, for two reasons:

1. They are not symmetric on {math}`\theta_1, \theta_2`
2. They rely on an arbitrary choice of {math}`F_+, F_\times`

In the non-precessing case, we can introduce a _distance_ between points of the manifold as:

```{math}
d^2(\theta_1, \theta_2) = 1 - \sqrt{(\hat{h}_+(\theta_1)|\hat{h}_+(\theta_2))^2 + [\hat{h}_+(\theta_1)|\hat{h}_+(\theta_2)]^2 }
```

The distance does not take into account the cross polarization and it is equivalent to setting {math}`F_\times = 0` in the standard match.
Indeed, for non-precessing signals {math}`\tilde{h}_+ = i \tilde{h}_\times` (hence {math}`(\hat{h}_+ | \hat{h}_\times) = 0` and {math}`[\hat{h}_+ | \hat{h}_\times] = 1`) and the standard match does not depend on the value of the antenna patterns: this allows to set {math}`F_\times = 0` and symmetrize the expression consistently.

For the general case, the distance above is not suitable and a more general distance definition should be worked out. This is work in progress: for the moment, one can use the distance above to place templates. However, there is a great risk that the results will be unreliable.

## A metric distance between templates

The distance above can be approximated by a metric distance {math}`d_\text{metric}`. A metric distance is a bilinear form, represented by a matrix  {math}`M_{ij}(\theta)` defined at each point of the space.

```{math}
d^2_\text{metric}(\theta_1, \theta_2) = M_{ij}(\frac{\theta_1+\theta_2}{2}) \Delta\theta_i \Delta\theta_j
```
where {math}`\Delta\theta = \theta_1 - \theta_2`.

The matrix {math}`M_{ij}(\theta)` can is built in such a way that the metric distance approximates the "true" distance:

```{math}
d_\text{metric}(\theta_1, \theta_2) \simeq d(\theta_1, \theta_2)
```

The metric can be computed at any point {math}`\theta` of the manifold using {func}`mbank.metric.cbc_metric.get_metric`; once can also compute the match {math}`\mathcal{M}_\text{metric} = 1 - d^2_\text{metric}` approximated by the metric by calling {func}`mbank.metric.cbc_metric.metric_match`

`mbank` heavily relies on such metric approximation to enourmously speed up the generation (and validation) of a template bank. 
