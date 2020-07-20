<!--
<script src="http://vjs.zencdn.net/4.0/video.js"></script>
-->

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<script type="text/javascript"> 
      // Show button
      function look(type){ 
      param=document.getElementById(type); 
      if(param.style.display == "none") param.style.display = "block"; 
      else param.style.display = "none" 
      } 
</script> 

<style>
.page {
  width: calc(100%);
}
</style>

# Diet deep generative audio models with structured lottery

**This website is still under construction. We keep adding new results, so please come back later if you want more.**

This website presents additional material and experiments around the paper *Diet deep generative audio models with structured lottery*.

Deep learning models have provided extremely successful solutions in most audio application fields. 
However, the high accuracy of these models comes at the expense of a tremendous computation cost. 
This aspect is almost always overlooked in evaluating the quality of proposed models. 
However, models should not be evaluated without taking into account their complexity. 
This aspect is especially critical in audio applications, which heavily relies on specialized embedded hardware with real-time constraints.

In this paper, we build on recent observations that deep models are highly overparameterized, by studying the *lottery ticket hypothesis* on deep generative audio models. 
This hypothesis states that extremely efficient small sub-networks exist in deep models and would provide higher accuracy than larger models if trained in isolation. 
However, lottery tickets are found by relying on unstructured *masking*, which means that resulting models do not provide any gain in either disk size or inference time. 
Instead, we develop here a method aimed at performing *structured trimming*. 
We show that this requires to rely on *global* selection and introduce a specific criterion based on mutual information.

First, we confirm the surprising result that *smaller models* provide *higher accuracy* than their large counterparts. 
We further show that we can remove up to **95%** of the model weights without significant degradation in accuracy. 
Hence, we can obtain very light models for generative audio across popular methods such as *Wavenet*, *SING* or *DDSP*, that are up to 100 times smaller with commensurate accuracy. 
We study the theoretical bounds for embedding these models on Raspberry Pi and Arduino, and show that we can obtain generative models on CPU with equivalent quality as large GPU models. 
Finally, we discuss the possibility of implementing deep generative audio models on embedded platforms.

**Examples contents**
  * [Audio reconstruction](#audio-reconstruction)
  * [Model specific results](#model-specific-results)

**Code and implementation**
  * [Source code](#code)

**Additional details**
  * [Mathematical appendix](#mathematical-appendix)
  * [Models architecture](#models-details)


## Audio reconstruction

Our first experiment consists in evaluating the reconstruction ability of our lightweight models. 

## Model specific results

Here we detail the results from the paper by separating the evaluation for each of the models.

## Code

The full open-source code is currently available on the corresponding [GitHub repository](https://github.com/acids-ircam/lottery_generative). 
Code has been developed with `Python 3.7`. It should work with other versions of `Python 3`, but has not been tested. 
Moreover, we rely on several third-party libraries that can be found in the README.

The code is mostly divided into two scripts `train.py` and `evaluate.py`. 
The first script `train.py` allows to train a model from scratch as described in the paper. 
The second script `evaluate.py` allows to generate the figures of the papers, and also all the supporting additional materials visible on this current page.

## Mathematical appendix

## Models details

### Wavenet

### SING

### DDSP

### Optimization