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

# Ultra-light deep MIR by trimming lottery tickets

**This website is still under construction. We keep adding new results, so please come back later if you want more.**

This website presents additional material and experiments around the paper *Ultra-light deep MIR by trimming lottery tickets*.

Current state-of-art results in Music Information Retrieval are largely dominated by deep learning approaches. 
These provide unprecedented accuracy across all discriminative tasks. 
However, the consistently overlooked downside of these models is their stunningly massive complexity, which seems concomitantly crucial to their success. 

In this paper, we address this issue by developing a new model pruning method based on the *lottery ticket hypothesis*. 
We modify the original approach to allow for explicitly removing parameters, through *structured trimming* of entire units, instead of simply masking individual weights. 
This leads to models which are effectively lighter in terms of size, memory and number of operations.

We show that our proposal can remove up to 90% of the model parameters without loss of accuracy, 
leading to ultra-light deep MIR models. We confirm the surprising result that, at smaller compression ratios (removing up to 80% of a network), lighter models consistently outperform their heavier counterparts. 
We exhibit these results on a large array of MIR tasks including *audio classification*, *pitch recognition*, *chord extraction*, *drum transcription* and *onset estimation*. 
The resulting ultra-light deep learning models for MIR can run on CPU, and can even fit on embedded devices with minimal degradation of accuracy. 

**Examples contents**
  * [Task specific results](#task-specific-results)

**Code and implementation**
  * [Source code](#code)

**Additional details**
  * [Mathematical appendix](#mathematical-appendix)
  * [Models architecture](#models-details)


## Audio reconstruction

Our first experiment consists in evaluating the reconstruction ability of our lightweight models. 

## Task specific results

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

### Optimization