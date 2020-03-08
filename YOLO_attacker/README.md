

## YOLO v3 attacker

two types of yolo v3 attacker.

thanks very much for NeuralSec/Daedalus-attack 's contribution. Our code is modified based on their work.
> https://github.com/NeuralSec/Daedalus-attack

---- 84 and his son.

### NMS_attacker

#### idea

Most of object detection algorithms use Non-Maximum Suppression to filter detection boxes. NMS has two fixed thresholds. NMS calculates box conficence and selects them through the first threshold, then calcualte Intersection over Union (IoU) between detection boxes and filter boxes again through the other fixed threshold. That process gives us something about adversarial attack to image

We consider three elemental schemes to break NMS. First, for YOLO-v3, it is necessary to make most of the bounding boxes survive in the first round of filtering that discards boxes based on the box confidences. Hence, we need to maximise the box confidences. Second, we can directly minimise the IoU for each pair of boxes. Alternatively, we can minimise the expectation of the box dimension over all the bounding boxes, and maximise the expectation of the Euclidean distance between box centres over all pairs of boxes.

#### some tips

we neet to control the input of YOLO-v3 between 0-1, which poses a box-constraint convex optimization question on loss function. So we take a tanh map, a pretty niubi method, on original image then add perturbation on tanh space. After that we take inverse tanh map to get image on real space.

#### loss functions

$$
\begin{equation}
\begin{aligned}
l2\_dist &= \sum(x - x_{adv}))^2 \\ \\
box\_score &= \frac{1}{box\_num}\sum (object\_score \times \max(class\_score) -1)^2 \\ \\
box\_shape &= \frac{1}{box\_num}\sum (width \times height)^2 \\ \\
loss &= l2\_dist + C \cdot (box\_score + box\_shape)
\end{aligned}
\end{equation}
$$





### Object_attacker

#### idea

we choose a more simple and natural way to attack yolo v3 model. That is, Controlling the prediction of adversarial samples to have just one box detection in the central of image. Also we can choose class of this detected box. We believe this attack method is natural because it is hard for people or automatic algorithms to find something wrong taking place on their OD model. On the contrary, NMS attacker seems to obvious due to so many detected boxes in the image.

For achieving this goal, we set three elements on loss function. First, we try to minimize object conficence of all detected box. Through first NMS filter we can discard this boxes because of their low object confidence. Second, we leave only one box that stays in the middle of image to have a high object confidence and class confidence where class is selected by user. Finally we control the box left to have specific width and height.

#### some tips

When training our adversarial examples, there exists some problems to reduce "loss_maybox" part of loss(see loss functions below). That's mainly because we use sigmoid function before having the final box width and height. To solve this sort of gradient vanishing issue, we set another optimizer to specificly optimize "loss_mybox". We run this optimizer every 10 epoches we run the general optimizer. 

#### loss functions

$$
\begin{equation}
\begin{aligned}
l2\_dist &= \sum(x - x_{adv}))^2 \\ \\
loss\_otherbox &= \frac{1}{box\_num}\sum (object\_score \times \max(class\_score))^2 \\ \\
loss\_mybox &= \ (1- object\_score \times choose\_class)^2 \\ \\
loss\_mysize &= \frac{1}{4} [ (x - x')^2+(y - y')^2+(h - h')^2+(w - w')^2] \\ \\
loss &= l2\_dist + C \cdot (loss\_otherbox + loss\_mybox + loss\_mysize)

\end{aligned}
\end{equation}
$$


### YOLO v3

Study Materials for Chinese learners

structure:
> https://blog.csdn.net/leviopku/article/details/82660381

definition of loss function:
> https://www.jianshu.com/p/86b8208f634f

K-means for anchor boxes
> https://blog.csdn.net/shiheyingzhe/article/details/83995213


