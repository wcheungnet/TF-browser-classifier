// timeline Array for training
const birdArray = [17, 18, 19, 20];
const bunnyArray = [105, 110, 142, 144, 147, 199, 222, 230, 253, 262, 264, 275, 50, 59, 63, 64, 67, 81, 87, 88, 93];
const butterflyArray = [107, 111, 136, 187, 188, 194, 203, 77, 78, 80];
const foxArray = [119, 153, 155, 162, 164, 210, 233];
const mouseArray = [117, 118, 119, 152, 162, 165, 210, 233];
const noCharArray = [100, 15, 225, 244, 24, 57, 8];
const squirrelArray = [120, 127, 137, 144, 145, 162, 188, 189, 195, 197, 201, 203, 204, 210, 233, 72];

const trainingDataSet1 = [{
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
        "class": "Fox",
        "timeline": foxArray
    },
    {
        "class": "Mouse",
        "timeline": mouseArray
    },
    {
        "class": "No Character",
        "timeline": noCharArray
    },
    {
        "class": "Squirrel",
        "timeline": squirrelArray
    },
];

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


let net;
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();

async function app(trainingData) {

    console('Loading mobileNet..');
    net = await mobilenet.load();

    const addExample = async (dataSet) => {
        for (const data of dataSet) {

            const classId = data.class;
            const timeArray = data.timeline;
            let i = 0;
            for (const t of timeArray) {
                i++;
                console(`Training ${classId} ${i}/${timeArray.length} <BR>Using first 5 minutes`);
                webcamElement.currentTime = t;
                const myActivation = net.infer(webcamElement, 'conv_preds');
                classifier.addExample(myActivation, classId);
                await sleep(.2);
            }
        }
    };

    await addExample(trainingData);

    webcamElement.currentTime = 0;
    webcamElement.play();

    while (true) {
        if (classifier.getNumClasses() > 0) {
            // Get the activation from mobilenet from the webcam.
            const activation = net.infer(webcamElement, 'conv_preds');

            // Get the most likely class and confidences from the classifier module.
            const result = await classifier.predictClass(activation);

            if (result.confidences[result.label] > 0.34) {
                console(`
                Prediction: ${trainingData[result.classIndex].class}<BR>
                Probability: ${Math.round(result.confidences[result.label] * 100)}%
                `);
            } else {
                console("Can't tell");
            }
        }
        await tf.nextFrame();
    }
}

async function sleep(time) {
    return new Promise((resolve) => {
        setTimeout(() => {
            return resolve(true);
        }, time * 1000);
    })
}

function console(string) {
    document.getElementById('console').innerHTML = string;
}

async function loadLocalImage(filename, sampleName) {
    return new Promise((resolve) => {
        try {
            const img = new Image()
            img.onload = () => {
                ctx.drawImage(img, 0, 0);
                let image = tf.browser.fromPixels(canvas);
                console.log(image);
                const activation = net.infer(img, 'conv_preds');
                console.log(activation);
                classifier.addExample(activation, sampleName);
                return resolve(null);
            }
            img.onerror = err => {
                throw err
            };
            img.src = filename;

        } catch (err) {
            console.log(err);
        }
    })
}

app(trainingDataSet2);