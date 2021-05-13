# Neural Style Transfer


Neural Style Transfer is a technique where style of one image is transfered to the other image.
NST uses CNN's to transfer style.

We'd be using pretrained VGG19 for our work.

Style can be computed by taking the co-relationship between the feature maps, it's computed using Gram Matrix

To transfer the style from one image to the content image we need to do two parallel operations while doing forward propagation
- Compute the content loss between the source image and the generated image
- Compute the style loss between the style image and the generated image
- Finally we need to compute the total loss

We need to reduce the total loss in order for the transfering the style from one image to another.

For calculating the gram matrix you could see this video :- <https://www.youtube.com/watch?v=DEK-W5cxG-g>


![](generated.png)