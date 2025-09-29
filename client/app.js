// MNIST Digit Classifier Frontend
class MNISTDrawingApp {
    constructor() {
        this.canvas = document.getElementById('drawingCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.lastDrawTime = 0;
        this.autoPredict = true;
        this.predictionTimeout = null;

        this.initializeCanvas();
        this.setupEventListeners();
        this.setupButtons();
    }

    initializeCanvas() {
        // Set up canvas for drawing (black background, white drawing)
        this.ctx.fillStyle = 'black';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.strokeStyle = 'white';
        this.ctx.lineWidth = 24; // pencil size
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
    }

    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseout', () => this.stopDrawing());

        // Touch events for mobile
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this.canvas.dispatchEvent(mouseEvent);
        });

        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this.canvas.dispatchEvent(mouseEvent);
        });

        this.canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            const mouseEvent = new MouseEvent('mouseup', {});
            this.canvas.dispatchEvent(mouseEvent);
        });
    }

    setupButtons() {
        document.getElementById('clearBtn').addEventListener('click', () => {
            this.clearCanvas();
        });

        document.getElementById('predictBtn').addEventListener('click', () => {
            this.makePrediction();
        });
    }

    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    startDrawing(e) {
        this.isDrawing = true;
        const pos = this.getMousePos(e);
        this.ctx.beginPath();
        this.ctx.moveTo(pos.x, pos.y);
        this.onDrawingChange();
    }

    draw(e) {
        if (!this.isDrawing) return;

        const pos = this.getMousePos(e);
        this.ctx.lineTo(pos.x, pos.y);
        this.ctx.stroke();
        this.onDrawingChange();
    }

    stopDrawing() {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.onDrawingChange();
        }
    }

    onDrawingChange() {
        this.lastDrawTime = Date.now();
        document.getElementById('predictBtn').disabled = false;

        // Clear any existing prediction timeout
        if (this.predictionTimeout) {
            clearTimeout(this.predictionTimeout);
        }

        // Set new timeout for auto-prediction
        if (this.autoPredict) {
            this.predictionTimeout = setTimeout(() => {
                this.makePrediction();
            }, 250); // Choose prediction delay [milliseconds]
        }
    }

    clearCanvas() {
        this.ctx.fillStyle = 'black';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.clearPrediction();
        document.getElementById('predictBtn').disabled = true;

        // Clear any pending prediction
        if (this.predictionTimeout) {
            clearTimeout(this.predictionTimeout);
        }
    }

    clearPrediction() {
        // Reset all digit boxes
        document.querySelectorAll('.digit-box').forEach(box => {
            box.classList.remove('predicted');
        });

        // Reset status
        this.updateStatus('Draw a digit to get started', '');
        document.getElementById('confidenceInfo').textContent = '';
    }

    // Convert canvas to 28x28 grayscale image
    canvasTo28x28() {
        // Create a temporary canvas for resizing
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = 28;
        tempCanvas.height = 28;

        // Draw scaled version
        tempCtx.fillStyle = 'black';
        tempCtx.fillRect(0, 0, 28, 28);
        tempCtx.drawImage(this.canvas, 0, 0, 280, 280, 0, 0, 28, 28);

        // Get image data
        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        const pixels = imageData.data;

        // Convert to grayscale and normalize to 0-1 range
        const grayscale = [];
        for (let i = 0; i < pixels.length; i += 4) {
            // Convert RGBA to grayscale (white drawing on black background)
            const r = pixels[i];
            const g = pixels[i + 1];
            const b = pixels[i + 2];
            const gray = (r + g + b) / 3;

            // Already in correct format: white digit on black background
            // Normalize to 0-1 range
            const normalized = gray / 255.0;
            grayscale.push(normalized);
        }

        return grayscale;
    }

    async makePrediction() {
        const apiUrl = document.getElementById('apiUrl').value.trim();
        if (!apiUrl) {
            this.updateStatus('Please enter an API URL', 'error');
            return;
        }

        try {
            this.updateStatus('Processing image and making prediction...', 'processing');

            // Convert canvas to 28x28 pixel array
            const imageArray = this.canvasTo28x28();

            // Make API call
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: imageArray
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`API Error (${response.status}): ${errorText}`);
            }

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            // Display prediction
            this.displayPrediction(result);

        } catch (error) {
            console.error('Prediction error:', error);
            this.updateStatus(`Error: ${error.message}`, 'error');
        }
    }

    displayPrediction(result) {
        // Clear previous predictions
        this.clearPrediction();

        // Highlight predicted digit
        const predictedDigit = result.prediction;
        const digitBox = document.querySelector(`[data-digit="${predictedDigit}"]`);
        if (digitBox) {
            digitBox.classList.add('predicted');
        }

        // Update status
        const confidence = (result.confidence * 100).toFixed(1);
        this.updateStatus(
            `Predicted: ${predictedDigit} (${confidence}% confidence)`,
            'success'
        );

        // Show detailed confidence info
        const confidenceInfo = document.getElementById('confidenceInfo');
        if (result.probabilities) {
            const probInfo = result.probabilities
                .map((prob, idx) => `${idx}: ${(prob * 100).toFixed(1)}%`)
                .join(' | ');
            confidenceInfo.textContent = `All probabilities: ${probInfo}`;
        }
    }

    updateStatus(message, type) {
        const statusDiv = document.getElementById('status');
        statusDiv.textContent = message;
        statusDiv.className = `status ${type}`;
    }

}

// Initialize the app when page loads
document.addEventListener('DOMContentLoaded', () => {
    const app = new MNISTDrawingApp();

    // Global reference for debugging
    window.mnistApp = app;

    console.log('MNIST Drawing App initialized');
    console.log('Available methods:', Object.getOwnPropertyNames(Object.getPrototypeOf(app)));
});