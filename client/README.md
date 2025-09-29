# MNIST Drawing Client

A minimalistic frontend for testing the MNIST digit classification model deployed in the MLOps workshop.

## Features

- **Interactive Drawing Canvas**: Draw digits with mouse or touch
- **Automatic Prediction**: Automatically predicts 1.25 seconds after you stop drawing
- **28x28 Export**: Converts drawings to proper MNIST format (784 pixels)
- **Visual Feedback**: Highlights predicted digit in the 0-9 display
- **API Integration**: Works with both local FastAPI and Modal deployments

## How to Use

### 1. Start a Local Server
```bash
# Simple Python server
python3 -m http.server 8080

# Or use any other local server
```

### 2. Open in Browser
```
http://localhost:8080
```

### 3. Configure API Endpoint
- **Part 1 (Local FastAPI)**: `http://localhost:8000/predict`
- **Part 4 (Modal)**: `https://username--mnist-classifier-predict.modal.run`

### 4. Draw and Test
1. Draw a digit (0-9) in the canvas
2. Wait 1.25 seconds for automatic prediction
3. See the predicted digit light up below
4. Check confidence scores

## Technical Details

### Drawing to Prediction Pipeline
1. **Canvas Drawing**: 280x280 pixel canvas with black ink
2. **Auto-trigger**: 1250ms delay after last drawing action
3. **Image Processing**:
   - Scale down to 28x28 pixels
   - Convert to grayscale
   - Invert colors (black ink → white digit on black background)
   - Normalize to 0-1 range
4. **API Call**: POST request with 784-pixel array
5. **Result Display**: Highlight predicted digit with confidence

### Canvas Processing
```javascript
// Canvas (280x280) → 28x28 → 784 pixel array → API → Prediction
canvasTo28x28() // Converts drawing to MNIST format
```

### API Format
```javascript
// Request
{
  "image": [0.0, 0.0, 0.1, 0.8, ...] // 784 pixel values (0-1 range)
}

// Response
{
  "prediction": 7,
  "class_name": "7",
  "confidence": 0.999,
  "probabilities": [0.001, 0.001, ..., 0.999, 0.001]
}
```

## Troubleshooting

### Common Issues
- **CORS errors**: Make sure your API server allows cross-origin requests
- **No prediction**: Check API URL and ensure backend is running
- **Poor accuracy**: Try drawing larger, centered digits
- **Touch issues**: Mobile touch events are supported

### Testing Tips
- Use the "Test with Sample" button to verify API connectivity
- Draw digits similar to handwritten style (not printed fonts)
- Make sure digits fill most of the canvas area
- Clear canvas between different digit tests

## Files

- `index.html` - Main HTML interface
- `app.js` - Drawing and prediction logic
- `README.md` - This documentation

## Integration with Workshop Parts

- **Part 1**: Test with local FastAPI server
- **Part 2**: Test with Kubernetes deployed service (via port-forward)
- **Part 3**: Test with cloud deployed endpoints
- **Part 4**: Test with Modal serverless deployment

This client provides an interactive way to validate that your ML model deployment is working correctly across all parts of the workshop!