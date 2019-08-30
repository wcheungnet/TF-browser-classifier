# TF-browser-classifier

## Classifier character in a video Big Buck Bunny.

[https://wcheungnet.github.io/TF-browser-classifier/](https://wcheungnet.github.io/TF-browser-classifier/)

I use random keyframe in first 5 min of the video.

After training, it will start the prediction!


### Create a sample array with keyframe in second

For Example, the bird appear on 17, 18, 19 and 20 second during the video play.

```js

// timeline Array for training
const birdArray = [17, 18, 19, 20];
const bunnyArray = [105, 110, 142, 144, 147, 199, 222, 230, 253, 262, 264, 275, 50, 59, 63, 64, 67, 81, 87, 88, 93];
const butterflyArray = [107, 111, 136, 187, 188, 194, 203, 77, 78, 80];
const foxArray = [119, 153, 155, 162, 164, 210, 233];
const mouseArray = [117, 118, 119, 152, 162, 165, 210, 233];
const noCharArray = [100, 15, 225, 244, 24, 57, 8];
const squirrelArray = [120, 127, 137, 144, 145, 162, 188, 189, 195, 197, 201, 203, 204, 210, 233, 72];

// Create training data set
const trainingDataSet2 = [{
    "class": "Bird",
    "timeline": birdArray
},
{
    "class": "Bunny",
    "timeline": bunnyArray
},
{
    "class": "Butterfly",
    "timeline": butterflyArray
},
{
    "class": "Evil brothers",
    "timeline": [...foxArray, ...mouseArray, ...squirrelArray]
},
{
    "class": "No Character",
    "timeline": noCharArray
}
];

```


### Train the classifier with data set

This code is used for move video cursor to t seconds.

```js
videoElement.currentTime = t;
```

Add it to our AddExample function

```js

const addExample = async (dataSet) => {
    for (const data of dataSet) {

        const classId = data.class;
        const timeArray = data.timeline;
        let i = 0;
        for (const t of timeArray) {
            i++; 
            console(`Training ${classId} ${i}/${timeArray.length} <BR>Using first 5 minutes`);
            videoElement.currentTime = t;
            const myActivation = net.infer(videoElement, 'conv_preds');
            classifier.addExample(myActivation, classId);
            await sleep(.2);
        }
    }
};

```

Rest of the code is similar with the tutorial

