"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import random
import google.generativeai as genai

import time 
from transformers import BartTokenizer, BartForConditionalGeneration


# from utils import *
openai_api_key = 

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

try:
    genai.configure(api_key=openai_api_key)
except KeyError:
    print("오류: GEMINI_API_KEY 환경 변수를 설정해주세요.")
    # 또는 하드코딩 (보안상 권장되지 않음)
    # genai.configure(api_key="YOUR_GEMINI_API_KEY")

def temp_sleep(seconds=0.5):
  time.sleep(seconds)

def gemini_request(prompt, model_name = "gemini-pro"):
    """
    Gemini 모델에 단일 프롬프트를 보내고 응답을 받습니다.

    Args:
        prompt (str): 모델에 전달할 프롬프트.
        model_name (str): 사용할 모델 이름.

    Returns:
        str: 모델의 텍스트 응답 또는 오류 메시지.
    """
    temp_sleep()
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        # 안전 등급으로 인해 응답이 차단되었는지 확인
        if not response.parts:
            # response.prompt_feedback.block_reason.name을 통해 차단 이유 확인 가능
            print("Gemini API Warning: 응답이 안전 설정에 의해 차단되었습니다.")
            return "GEMINI_RESPONSE_BLOCKED"
        return response.text
    except Exception as e:
        print(f"Gemini API ERROR: {e}")
        return "Gemini API ERROR"

# ============================================================================
# #####################[SECTION 1: GEMINI STRUCTURE] ######################
# ============================================================================

def gemini_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat: int = 3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose: bool = False):
    """
    Gemini에 JSON 형식의 출력을 요청하고, 결과를 검증하고 정리합니다.

    Args:
        prompt (str): 기본 프롬프트.
        example_output (str): JSON 출력의 예시.
        special_instruction (str): JSON 형식에 대한 추가 지침.
        repeat (int): 유효한 응답을 얻기 위한 최대 시도 횟수.
        func_validate (function): 응답을 검증하는 함수.
        func_clean_up (function): 응답을 정리하는 함수.
        verbose (bool): 디버깅 정보 출력 여부.

    Returns:
        성공 시 정리된 응답, 실패 시 False.
    """
    full_prompt = f'"""\n{prompt}\n"""\n'
    full_prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    full_prompt += "Example output json:\n"
    full_prompt += '{"output": "' + str(example_output) + '"}'

  
    if verbose:
        print("GEMINI PROMPT")
        print(full_prompt)

    for i in range(repeat):
        try:
            curr_gemini_response = gemini_request(full_prompt).strip()
            end_index = curr_gemini_response.rfind('}') + 1
            curr_gemini_response = curr_gemini_response[:end_index]
            curr_gemini_response = json.loads(curr_gemini_response)["output"]
            
            if func_validate(curr_gemini_response, prompt=prompt): 
                return func_clean_up(curr_gemini_response, prompt=prompt)

            if verbose:
                print(f"---- repeat count: {i}\n", json_response)
                print (curr_gemini_response)
                print("~~~~")

        except:
            pass

    return False

# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def gemini_text_completion_request(prompt, generation_params, model_name = "gemini-pro"):
    """
    세부 파라미터를 설정하여 Gemini에 텍스트 완성을 요청합니다.

    Args:
        prompt (str): 모델에 전달할 프롬프트.
        generation_params (dict): 생성 관련 파라미터 (temperature, max_output_tokens 등).
        model_name (str): 사용할 모델 이름.

    Returns:
        str: 모델의 텍스트 응답 또는 오류 메시지.
    """
    temp_sleep()
    try:
        # Gemini API 파라미터에 맞게 변환
        config = genai.types.GenerationConfig(
            temperature=generation_params.get("temperature", 0.7),
            max_output_tokens=generation_params.get("max_tokens"),
            top_p=generation_params.get("top_p"),
            stop_sequences=generation_params.get("stop")
        )
        # 참고: frequency_penalty, presence_penalty는 Gemini API에서 직접 지원하지 않음

        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt, generation_config=config)
        return response.text
    except :
        print ("TOKEN LIMIT EXCEEDED")
        return "TOKEN LIMIT EXCEEDED"

def safe_generate_response(prompt,
                           generation_params,
                           repeat: int = 5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose: bool = False):
    """
    응답을 받고 검증/정리하는 과정을 반복하는 안전한 생성 함수.
    """
    if verbose:
        print(prompt)

    for i in range(repeat):
        curr_gemini_response = gemini_text_completion_request(prompt, generation_params)
        
        if func_validate(curr_gemini_response, prompt=prompt):
            return func_clean_up(curr_gemini_response, prompt=prompt)
    
        if verbose:
            print ("---- repeat count: ", i, curr_gemini_response)
            print (curr_gemini_response)
            print ("~~~~")
            
    return fail_safe_response

def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()

def get_embedding(text, model: str = "models/embedding-001"):
    """
    주어진 텍스트의 임베딩 벡터를 생성합니다.

    Args:
        text (str): 임베딩할 텍스트.
        model (str): 사용할 임베딩 모델 이름.

    Returns:
        list: 텍스트의 임베딩 벡터.
    """
    text = text.replace("\n", " ")
    if not text:
        text = "this is blank"
    # 2. 텍스트 토큰화
    # 모델이 처리할 수 있도록 텍스트를 숫자 ID로 변환하고 PyTorch 텐서로 만듭니다.
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

    # 3. 모델을 통해 문맥적 임베딩 추출
    # 기울기 계산을 비활성화하여 메모리 사용량과 계산 속도를 향상시킵니다.
    with torch.no_grad():
        outputs = bart(**inputs)

    # 4. 평균 풀링(Mean Pooling) 수행
    # 마지막 은닉 상태(last_hidden_state)에서 패딩 토큰을 제외하고
    # 모든 토큰의 임베딩 벡터의 평균을 계산하여 문장 전체를 대표하는 단일 벡터를 만듭니다.
    attention_mask = inputs['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    
    mean_pooled_embedding = sum_embeddings / sum_mask
    
    # [1, 1024] 형태의 텐서에서 [1024] 벡터만 반환
    return mean_pooled_embedding.squeeze()


    result = genai.embed_content(model=model, content=text)
    return result['embedding']


if __name__ == '__main__':

    # Gemini API 파라미터 설정
    generation_params = {
        "temperature": 0,
        "max_tokens": 50, # Gemini에서는 max_output_tokens로 변환됨
        "top_p": 1,
        "stop": ['"'] # Gemini에서는 stop_sequences로 변환됨
    }
    
    curr_input = ["driving to a friend's house"]
    prompt_lib_file = "v1/test_prompt_July5.txt"
    prompt = generate_prompt(curr_input, prompt_lib_file)

    def __func_validate(gemini_response, **kwargs):
        response_str = gemini_response.strip()
        if len(response_str) <= 1:
            return False
        if len(response_str.split(" ")) > 1:
            return False
        return True

    def __func_clean_up(gemini_response, **kwargs):
        return gemini_response.strip()

    output = safe_generate_response(prompt,
                                    generation_params,
                                    repeat=5,
                                    fail_safe_response="rest",
                                    func_validate=__func_validate,
                                    func_clean_up=__func_clean_up,
                                    verbose=True)

    print("\n--- Final Output ---")
    print(output)
    
    # 임베딩 함수 테스트
    embedding_vector = get_embedding("Hello world")
    print("\n--- Embedding Vector (first 5 dimensions) ---")
    print(embedding_vector[:5])
