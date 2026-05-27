import json
import re
import traceback
import os
from storyteller.llm_call.openai_llm import call_openai
from storyteller.llm_call.prompt_factory import get_prompt
from typing import Dict, Any

def extract_json_from_text(text):
    """Extract JSON object from text"""
    try:
        # First try to parse the entire text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If failed, try to find JSON block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                json_str = json_match.group(1).strip()
                return json.loads(json_str)
            except:
                pass
                
        # Try to find JSON object enclosed in brackets
        json_match = re.search(r'(\{[\s\S]*\})', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
                
        print(f"❌ Failed to extract JSON from text: {text[:100]}...")
        return None

def get_prompt_content(template_name, template_args):
    """Read template file directly and replace variables in it to avoid Python string formatting issues"""
    # Get template file path
    template_path = os.path.join("storyteller", "templates", f"{template_name}.txt")
    
    # Read template content
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except Exception as e:
        print(f"❌ Failed to read template file: {str(e)}")
        return None
    
    # Replace variables in template
    for key, value in template_args.items():
        placeholder = "{" + key + "}"
        template_content = template_content.replace(placeholder, str(value))
    
    return template_content

def evaluate_report(
    dataset_context: str, 
    query: str, 
    md_report: str, 
    report_image: str = None,
    llm_kwargs: Dict[str, Any] = None,
    max_retries: int = 3
) -> float:
    """
    Evaluate data visualization report quality
    
    Args:
        dataset_context: Dataset context
        query: User query
        md_report: Report Markdown content
        report_image: Base64 encoded report screenshot (optional, not used in current implementation)
        llm_kwargs: LLM call parameters
        max_retries: Maximum retry attempts
    
    Returns:
        float: Weighted score (0-10)
    """
    # Build prompt (prepare outside retry loop to avoid repeated construction)
    prompt_args = {
        "DATASET_CONTEXT": dataset_context,
        "QUERY": query,
        "REPORT": md_report,
        "REPORT_IMAGE": ""  # Image not included
    }
    
    # Use custom function to get template content to avoid formatting issues
    prompt = get_prompt_content("report_evaluation", prompt_args)
    if not prompt:
        print("❌ Failed to get template content")
        return 5.0  # Default medium score
        
    # Retry mechanism
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                print("📝 Evaluating report using generic API...")
            else:
                print(f"🔄 Retry {attempt + 1} for report evaluation...")
            
            # Call API for evaluation
            responses = call_openai(prompt, **(llm_kwargs or {}))
            if not responses:
                print("⚠️ API returned no valid response")
                if attempt == max_retries - 1:  # Last attempt
                    return 5.0  # Default medium score
                continue
                
            response_text = responses[0].strip()
            print("✅ Successfully received evaluation response")
            
            # Output raw response for debugging
            print(f"\n📝 Evaluation response (first 200 chars):\n{response_text[:200]}...")
            
            # Handle possible markdown format
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "")
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "")
            response_text = response_text.strip()
            
            # Try to parse JSON
            result = None
            
            # First try to parse JSON directly
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # If direct parsing failed, use enhanced JSON extraction method
                print("⚠️ Direct JSON parsing failed, trying to extract JSON...")
                result = extract_json_from_text(response_text)
                
            # Check if JSON was successfully parsed
            if not result:
                print(f"❌ Attempt {attempt + 1}: Failed to extract valid JSON from response")
                if attempt == max_retries - 1:  # Last attempt
                    print("❌ All retries failed, returning default score")
                    return 5.0  # Default medium score
                continue  # Continue to retry
            
            # Verify all required keys exist
            required_keys = ["informativeness", "clarity_coherence", "visualization_quality", "narrative_quality"]
            missing_keys = [key for key in required_keys if key not in result]
            if missing_keys:
                print(f"❌ Attempt {attempt + 1}: Missing required evaluation dimensions: {missing_keys}")
                if attempt == max_retries - 1:  # Last attempt
                    print("❌ All retries failed, returning default score")
                    return 5.0  # Default medium score
                continue  # Continue to retry
            
            # Ensure scores are numeric types
            try:
                for key in required_keys:
                    if not isinstance(result[key]["score"], (int, float)):
                        result[key]["score"] = float(result[key]["score"])
            except (ValueError, TypeError, KeyError) as e:
                print(f"❌ Attempt {attempt + 1}: Error converting score to number: {str(e)}")
                if attempt == max_retries - 1:  # Last attempt
                    print("❌ All retries failed, returning default score")
                    return 5.0
                continue  # Continue to retry
                
            # If we get here, parsing succeeded
            print(f"✅ Attempt {attempt + 1} successfully parsed JSON")
            
            # Calculate weighted score - all dimensions have equal weight (25%)
            weighted_score = (
                0.3 * result["informativeness"]["score"] +
                0.3 * result["clarity_coherence"]["score"] +
                0.2 * result["visualization_quality"]["score"] +
                0.2 * result["narrative_quality"]["score"]
            )
            
            # Print evaluation results
            print("\n📊 Report Evaluation Results:")
            print(f"- Informativeness (30%): {result['informativeness']['score']}/10")
            print(f"  Rationale: {result['informativeness']['rationale'][:200]}...")
            
            print(f"\n- Clarity and Coherence (30%): {result['clarity_coherence']['score']}/10")
            print(f"  Rationale: {result['clarity_coherence']['rationale'][:200]}...")
            
            print(f"\n- Visualization Quality (30%): {result['visualization_quality']['score']}/10")
            print(f"  Rationale: {result['visualization_quality']['rationale'][:200]}...")
            
            print(f"\n- Narrative Quality (30%): {result['narrative_quality']['score']}/10")
            print(f"  Rationale: {result['narrative_quality']['rationale'][:200]}...")
            
            print(f"\n✨ Weighted Total Score: {weighted_score:.2f}/10")
            
            return round(weighted_score, 2)
            
        except Exception as e:
            print(f"❌ Error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:  # Last attempt
                print("❌ All retries failed, returning default score")
                traceback.print_exc()  # Print detailed error stack
                return 5.0
            # Continue to retry
    
    # Theoretically won't reach here, but as a safety net
    return 5.0
    