from ctransformers import AutoModelForCausalLM
import sys
llm = AutoModelForCausalLM.from_pretrained('TheBloke/Llama-2-7B-Chat-GGML', model_file = 'llama-2-7b-chat.ggmlv3.q4_K_S.bin' )

def main(user_input_text):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<>\n", "\n<>\n\n"
    instruction="You Are Helpful AI language Model that will reply to Human"
    prompt=B_INST+instruction+E_INST+user_input_text
    for text in llm(prompt, stream=True):
        print(text, end="")
if __name__ == "__main__":
    # read the command line arguments
    args = sys.argv[1:]
    user_input_text =args[0]
    main(user_input_text)