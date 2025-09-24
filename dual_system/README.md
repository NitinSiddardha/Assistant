# Dual System LLM Router

A smart Flask application that automatically routes user prompts to the most appropriate AI model based on complexity assessment. **Now with a cleaner, more efficient architecture!**

## 🚀 **New Architecture**

### **🔄 Evaluation & Fast Responses**: Google Gemini Flash
- **Fast and cost-effective** for both evaluation and simple responses
- **Generous free tier** - no more rate limit issues
- **Excellent for**: Simple Q&A, basic tasks, prompt assessment

### **🧠 Complex Tasks**: OpenAI GPT-4
- **Powerful reasoning** for complex mathematical and analytical tasks
- **Excellent for**: Advanced math, complex reasoning, detailed explanations

## **Why This Architecture?**

1. **No More Rate Limits**: Gemini Flash has much higher free tier limits
2. **Cost Effective**: Gemini Flash is cheaper for simple tasks
3. **Best of Both Worlds**: Fast responses + powerful reasoning
4. **Simpler Fallback**: Natural redundancy between different APIs

## Environment Variables

Create a `.env` file in the project root with:

```bash
# Required API Keys
GOOGLE_API_KEY=your_google_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Flask secret key for sessions
FLASK_SECRET_KEY=your_secret_key_here

# Model Configuration (New Default Architecture)
FAST_MODEL_ID=gemini-1.5-flash          # Default: Gemini Flash for fast responses
BIG_MODEL_ID=gpt-4                      # Default: OpenAI GPT-4 for complex tasks
EVAL_MODEL_ID=gemini-1.5-flash          # Default: Gemini Flash for evaluation

# Alternative Configurations:
# FAST_MODEL_ID=gpt-3.5-turbo           # Use OpenAI for fast responses
# BIG_MODEL_ID=gemini-1.5-pro           # Use Gemini Pro for complex tasks

# Optional: Routing Thresholds
PROMPT_LENGTH_THRESHOLD=400             # Default: 400 characters
CONFIDENCE_THRESHOLD=0.5                # Default: 0.5 (50%)
MAX_TOKENS_FAST=128                     # Default: 128 tokens
MAX_TOKENS_BIG=512                      # Default: 512 tokens
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

3. Create your `.env` file with the configuration above

4. Run the application:
```bash
python app.py
```

## How It Works

### 1. **Prompt Assessment**
Every user prompt is analyzed by Gemini Flash to determine:
- **Complexity** (0.0-1.0): How difficult the task is
- **Task Type**: Classification, Math, Code Generation, etc.
- **Stopword Percentage**: How much "filler" content vs. meaningful content

### 2. **Intelligent Routing**
Based on assessment, prompts are routed to:
- **Gemini Flash**: Simple questions, basic tasks, high confidence
- **OpenAI GPT-4**: Complex reasoning, advanced topics, low confidence

### 3. **Natural Fallback**
If one API fails, the other naturally handles the request:
- **Gemini Flash fails** → OpenAI handles it
- **OpenAI fails** → Gemini Flash handles it
- **No complex fallback logic needed**

## 🔄 **Easy Model Switching**

Want to try different models? Just change your `.env` file:

```bash
# Example 1: All Gemini models
FAST_MODEL_ID=gemini-1.5-flash
BIG_MODEL_ID=gemini-1.5-pro
EVAL_MODEL_ID=gemini-1.5-flash

# Example 2: All OpenAI models  
FAST_MODEL_ID=gpt-3.5-turbo
BIG_MODEL_ID=gpt-4
EVAL_MODEL_ID=gpt-3.5-turbo

# Example 3: Mixed approach (current default)
FAST_MODEL_ID=gemini-1.5-flash          # Gemini for speed
BIG_MODEL_ID=gpt-4                      # OpenAI for complexity
EVAL_MODEL_ID=gemini-1.5-flash          # Gemini for assessment
```

## API Usage

- **Evaluation Model**: Gemini Flash for prompt assessment
- **Fast Model**: Gemini Flash for simple, quick responses
- **Big Model**: OpenAI GPT-4 for complex, detailed responses

## Benefits of New Architecture

✅ **No more rate limit issues** with Gemini Flash  
✅ **Cost-effective** for simple tasks  
✅ **Powerful reasoning** for complex tasks  
✅ **Natural redundancy** between APIs  
✅ **Simpler code** - no complex fallback logic  
✅ **Better user experience** - always gets a response  

## Error Handling

The system includes comprehensive error handling for:
- API failures and quota limits
- JSON parsing errors
- Model routing issues
- Invalid responses
- Natural fallback between APIs
