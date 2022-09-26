Generating templates with normalizing flow
==========================================

A [normalizing flow](https://arxiv.org/abs/1912.02762) is a powerful Machine Learning model which allow to sample from a complicated distribution in an analytical way. Since the template placing can be cast into a sampling (+ rejection) problem, it is natural to use such model to improve the performance of the metric placement.
In this context, the flow acts both as a sampling model and metric interpolator, with great improvement in template placement performance.

This part of the code is still work-in-progress and the same applies to its documentation: coming soon :D
