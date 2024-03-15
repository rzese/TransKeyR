// Recupera gli elementi HTML tramite gli ID
const webcamButton = document.getElementById('webcam-button');
const predictButton = document.getElementById('predict-button');
const captureButton = document.getElementById('capture-button');
const webcamVideo = document.getElementById('webcam-video');
const webcamCanvas = document.getElementById('webcam-canvas');
const webcamPreview = document.getElementById('webcam-preview');
const errorMessageElement = document.getElementById('error-message');
let model = null;

// Logica per la gestione del modal della webcam (finestra popup)
var modal = document.getElementById("webcam-modal");
var span = document.getElementsByClassName("close")[0];

// Mostra il modal della webcam quando si clicca sul pulsante della webcam
webcamButton.onclick = () => {
    // Nasconde eventuali messaggi di errore
    if (errorMessageElement) {
        errorMessageElement.style.display = 'none';
    }
    // Mostra il modal
    modal.style.display = "block";
    // Avvia la webcam
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: { width: 256, height: 256 } })
            .then(stream => {
                webcamVideo.srcObject = stream;
                webcamVideo.onloadedmetadata = (e) => {
                    webcamVideo.play();
                    // Mostra il pulsante di scatto quando la webcam è attiva
                    captureButton.style.display = 'block';
                };
            });
    }
}

// Cattura un'immagine dalla webcam quando si clicca sul pulsante di scatto
captureButton.onclick = () => {
    const context = webcamCanvas.getContext('2d');
    context.drawImage(webcamVideo, 0, 0, 256, 256);
    imageData = webcamCanvas.toDataURL('image/png');
    // Mostra un'anteprima dell'immagine
    webcamPreview.src = imageData;
    webcamPreview.style.display = 'inline-block';
    predictButton.style.display = 'inline-block';
  }

// Chiude il modal quando si clicca sul pulsante di chiusura
span.onclick = function() {
    modal.style.display = "none";
    // Ferma la webcam
    if (webcamVideo.srcObject) {
        let stream = webcamVideo.srcObject;
        let tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        webcamVideo.srcObject = null;
    }
}
// Chiude il modal se si clicca fuori dal suo contenuto
window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
        if (webcamVideo.srcObject) {
            let stream = webcamVideo.srcObject;
            let tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            webcamVideo.srcObject = null;
        }
    }
}
// Cattura un'immagine dalla webcam quando si clicca sul video della webcam
webcamVideo.onclick = () => {
    const context = webcamCanvas.getContext('2d');
    context.drawImage(webcamVideo, 0, 0, 256, 256);
    imageData = webcamCanvas.toDataURL('image/png');
    // Mostra un'anteprima dell'immagine e mostra il pulsante per fare una previsione
    webcamPreview.src = imageData;
    webcamPreview.style.display = 'block';
    predictButton.style.display = 'block';
}

// Array dei nomi dei modelli
const modelNames = ['ResNet50', 'EfficentNetB1', 'YOLOv5', 'YOLOv8'];
let predictionContainer;
let predictionTextContainer;
// Logica per effettuare una previsione quando si clicca sul pulsante di previsione
predictButton.onclick = () => {
    // Rimuovi tutte le previsioni e le immagini esistenti
    const oldPredictions = document.querySelectorAll('.model-prediction');
    oldPredictions.forEach((element) => {
        element.remove();
    });
    // Reinizializza il contenitore delle previsioni creando un nuovo contenitore div per ospitare le previsioni del modello
    predictionContainer = document.createElement('div');
    predictionContainer.classList.add('model-prediction');
    const imageDataStr = imageData.split(',')[1]; // Rimuove "data:image/png;base64," dall'inizio dell'URL dell'immagine             
    const timestamp = Date.now(); // Ottieni il timestamp corrente
    const imageName = `webcam_image_${timestamp}.png`; // Genera un nome univoco per l'immagine
    // Invia una richiesta POST al server con l'immagine catturata
    fetch('/predict-webcam', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageDataStr }) // L'immagine viene convertita in una stringa e inviata come JSON
    })

    .then(response => response.json())  // Converte la risposta del server in un oggetto JSON
    .then(data => {
        // Gestisce la risposta del server
        // Qui, 'data' contiene le previsioni del modello e gli URL delle immagini processate
        
        // Verifica se ci sono già elementi di previsione nella pagina
        const predictionElements = document.getElementsByClassName('prediction-row');
        // Se esistono già elementi di previsione, aggiorna i loro valori
        if (predictionElements.length > 0) {
            for (let i = 0; i < data.predictions.length; i++) {
                predictionElements[i].children[1].textContent = data.predictions[i];
            }
            // Aggiorna le immagini esistenti
            const existingImages = document.querySelectorAll('.prediction-image');
            existingImages[0].src = data.image_url;  // Immagine originale
            existingImages[1].src = data.yolov5_url; // Immagine processata da YOLOv5
            existingImages[2].src = data.yolov8_url; // Immagine processata da YOLOv8            
            // Aggiungi il parametro di timestamp
            predictionTextContainer = document.querySelector('.prediction-text');
        } else {
            // Creare nuovi elementi di previsione se non ne esistono
            predictionContainer.classList.add('model-prediction');
            
            predictionTextContainer = document.createElement('div');
            predictionTextContainer.classList.add('prediction-text');
            predictionContainer.appendChild(predictionTextContainer);
            
            // Aggiungi le nuove righe di previsione
            for (let i = 0; i < data.predictions.length; i++) {
                addNewPredictionRow(predictionTextContainer, modelNames[i], data.predictions[i]);  
            }
            //Aggiunge le immagini originali
            const predictionImage = document.createElement('img');
            predictionImage.classList.add('prediction-image');
            predictionImage.src = data.image_url;
            predictionImage.alt = 'Immagine caricata';
            predictionContainer.appendChild(predictionImage);
            document.body.appendChild(predictionContainer);
            
        }
        // Aggiungi le immagini processate da YOLOv5 e YOLOv8
        addNewPredictionImage(predictionContainer, 'Predizione YOLOv5', data.yolov5_url);
        addNewPredictionImage(predictionContainer, 'Predizione YOLOv8', data.yolov8_url);
        
        // Chiudi il riquadro della webcam
        modal.style.display = "none";
        if (webcamVideo.srcObject) {
            let stream = webcamVideo.srcObject;
            let tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            webcamVideo.srcObject = null;
        }
        

    });
    // Funzioni helper per aggiungere nuove righe di previsione e immagini
    function addNewPredictionRow(container, label, value) {
        const predictionElement = document.createElement('div');
        predictionElement.classList.add('prediction-row');
        
        const predictionLabel = document.createElement('h3');
        predictionLabel.textContent = label + ':';
        predictionElement.appendChild(predictionLabel);
        
        const predictionValue = document.createElement('p');
        predictionValue.textContent = value;
        predictionElement.appendChild(predictionValue);
        
        container.appendChild(predictionElement);
    }

    function addNewPredictionImage(container, altText, src) {
        const imageContainer = document.createElement('div');
        imageContainer.classList.add('image-container');
        imageContainer.classList.add('image-container-6');  
    
        const predictionImage = document.createElement('img');
        predictionImage.classList.add('prediction-image');
        predictionImage.src = src;
        predictionImage.alt = altText;
    
        const imageText = document.createElement('p');
        imageText.textContent = altText;
    
        imageContainer.appendChild(imageText);
        imageContainer.appendChild(predictionImage);
    
        container.appendChild(imageContainer);
    }
    

}