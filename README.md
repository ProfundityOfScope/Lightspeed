# Lightspeed
This is some code to test out visualizations of realistic relativistic spaceflight. People have tried to do this to some extent in the past, but it tends to involve some simplications, such as assuming all stars have the same spectrum, or only calculating brightness and not color. A full treatment requires effectively calculating the spectrum of the entire sky and evolving that relativistically, then convolving with various color response functions and transforming into color spaces we're more familiar with. 

The end goal is to produce an animation as if one were flying straight "up" or "out" from the galactic plane. That should minimize the number of stars which need to be considered, which is important as the intent is to generate a realistic population of perhaps tens of millions of stars in the main sequence and various giant branches. It may be possible to optimize these calculations using something like a quadtree, that's mostly just a note for myself.
