"""
Aviation Question Generation System - Web API
Simple Flask API for question generation service.
"""

from flask import Flask, request, jsonify, render_template_string
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

app = Flask(__name__)

# Global model variables
model = None
tokenizer = None
device = None

# HTML template for simple web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Aviation Question Generator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            margin: 10px 0;
            resize: vertical;
        }
        button {
            background-color: #3498db;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            margin-top: 10px;
        }
        button:hover {
            background-color: #2980b9;
        }
        .questions {
            margin-top: 30px;
            padding: 20px;
            background-color: #ecf0f1;
            border-radius: 5px;
        }
        .question-item {
            background-color: white;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #3498db;
            border-radius: 3px;
        }
        .label {
            font-weight: bold;
            color: #2c3e50;
            margin-top: 15px;
            display: block;
        }
        .slider-container {
            margin: 20px 0;
        }
        .slider {
            width: 100%;
        }
        .slider-label {
            display: flex;
            justify-content: space-between;
            margin-top: 5px;
            font-size: 12px;
            color: #666;
        }
        .info {
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
        }
        .loading {
            text-align: center;
            color: #666;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>✈️ Aviation Question Generator</h1>
        
        <div class="info">
            <strong>How to use:</strong> Enter aviation-related text below, and the AI will generate relevant questions for training and assessment.
        </div>
        
        <label class="label" for="context">Aviation Context:</label>
        <textarea id="context" rows="6" placeholder="Enter aviation-related text here... For example: 'The altimeter is an instrument that measures the height of an aircraft above sea level by detecting changes in atmospheric pressure.'"></textarea>
        
        <div class="slider-container">
            <label class="label" for="numQuestions">Number of Questions: <span id="numQuestionsValue">3</span></label>
            <input type="range" id="numQuestions" class="slider" min="1" max="10" value="3" oninput="document.getElementById('numQuestionsValue').textContent = this.value">
        </div>
        
        <button onclick="generateQuestions()">Generate Questions</button>
        
        <div class="loading" id="loading">
            <p>Generating questions... Please wait.</p>
        </div>
        
        <div id="results"></div>
    </div>
    
    <script>
        async function generateQuestions() {
            const context = document.getElementById('context').value;
            const numQuestions = document.getElementById('numQuestions').value;
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            
            if (!context.trim()) {
                alert('Please enter some context text.');
                return;
            }
            
            loading.style.display = 'block';
            results.innerHTML = '';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        context: context,
                        num_questions: parseInt(numQuestions)
                    })
                });
                
                const data = await response.json();
                loading.style.display = 'none';
                
                if (data.success) {
                    displayResults(data.questions);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                loading.style.display = 'none';
                alert('Error generating questions: ' + error);
            }
        }
        
        function displayResults(questions) {
            const results = document.getElementById('results');
            let html = '<div class="questions"><h2>Generated Questions:</h2>';
            
            questions.forEach((q, index) => {
                html += `<div class="question-item">${index + 1}. ${q}</div>`;
            });
            
            html += '</div>';
            results.innerHTML = html;
        }
    </script>
</body>
</html>
"""


def load_model(model_path='./aviation_qg_model/checkpoint-epoch-3'):
    """Load the trained model."""
    global model, tokenizer, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_path}...")
    print(f"Using device: {device}")
    
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")


def generate_questions_from_context(context, num_questions=3):
    """Generate questions from context."""
    input_text = f"generate question: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=64,
            num_return_sequences=num_questions,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            early_stopping=True
        )
    
    questions = [
        tokenizer.decode(output, skip_special_tokens=True)
        for output in outputs
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_questions = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)
    
    return unique_questions


@app.route('/')
def index():
    """Render the main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/generate', methods=['POST'])
def generate():
    """API endpoint for question generation."""
    try:
        data = request.json
        context = data.get('context', '')
        num_questions = data.get('num_questions', 3)
        
        if not context:
            return jsonify({
                'success': False,
                'error': 'No context provided'
            }), 400
        
        if num_questions < 1 or num_questions > 10:
            return jsonify({
                'success': False,
                'error': 'Number of questions must be between 1 and 10'
            }), 400
        
        questions = generate_questions_from_context(context, num_questions)
        
        return jsonify({
            'success': True,
            'questions': questions,
            'context': context
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


def main():
    """Start the Flask application."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Aviation Question Generation API')
    parser.add_argument('--model-path', type=str, 
                       default='./aviation_qg_model/checkpoint-epoch-3',
                       help='Path to trained model')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the server on')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to run the server on')
    
    args = parser.parse_args()
    
    # Load model before starting server
    load_model(args.model_path)
    
    print(f"\n{'='*60}")
    print("Aviation Question Generation API Server")
    print(f"{'='*60}")
    print(f"Server starting on http://{args.host}:{args.port}")
    print(f"Open your browser and navigate to the URL above")
    print(f"{'='*60}\n")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
