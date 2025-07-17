import streamlit as st
import time
import re

from pathlib import Path
from io import StringIO

root = Path.cwd().parent.parent
the_verdict = root/"git"/"SWRG"/"Chapter_2"/"resources"/"the-verdict.txt"

def note(text: str):
    return f"\n 🔶 {text}"

def stream_data(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab            #1
        self.int_to_str = {i:s for s,i in vocab.items()}        #2

    def encode(self, text):         #3
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):         #4
        text = " ".join([self.int_to_str[i] for i in ids]) 

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)    #5
        return text

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int            #1
                        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)    #2
        return text

if "show_section_2_2" not in st.session_state:
    st.session_state.show_section_2_2 = False
if "show_section_2_3" not in st.session_state:
    st.session_state.show_section_2_3 = False
if "show_section_2_4" not in st.session_state:
    st.session_state.show_section_2_4 = False
if "show_section_real_world" not in st.session_state:
    st.session_state.show_section_real_world = False
if "verdict_loaded" not in st.session_state:
    st.session_state.verdict_loaded = False
if "verdict_content" not in st.session_state:
    st.session_state.verdict_content = ""
if "verdict_preprocess" not in st.session_state:
    st.session_state.verdict_preprocess = ""


st.sidebar.write("**Chapter 2: Working with text data**")
if st.sidebar.button("Introduction", type="tertiary"):
    st.write_stream(stream_data(text=note("During pretraining stage, LLMs process text one word at a time.")))
    st.write_stream(stream_data(text=note("Before we can implement and train LLMs, we need to prepare the training dataset.")))
    st.image(root/"SWRG"/"pipeline_full.png")
    
if st.sidebar.button("Section 2.1 - Understanding word embeddings", type="tertiary"):
    st.write_stream(stream_data(text=note("LLMs aren't capable of processing text as we can.")))
    st.write_stream(stream_data(text=note("To make the text compatible with mathematical operations used for training/inference\
                                          we need to represent words as continuous-valued vectors")))
    st.write_stream(stream_data(text=note("Converting data into vector format is referred to as embedding")))
    st.write_stream(stream_data(text=note("A mental model for an embedding matrix is a look-up table, a hash map, a dictionary")))
    st.image(root/"SWRG"/"embedding_image.png")
    st.write_stream(stream_data(text=note("The primary purpose of embeddings is to convert nonnumeric data into a format that\
                                          neural networks can process")))
    st.write_stream(stream_data(text=note("As seen in the image above, although word embeddings are most common,\
                                          you can also have embeddings for sentences (RAG), paragraphs (RAG), or documents")))
    st.write_stream(stream_data(text=note("Several algorithms and frameworks have been developed to generate word embeddings\
                                          with one of the earliest and most popular being Word2Vec.")))
    st.write_stream(stream_data(text=note("The main idea behind Word2Vec is that words appearing in similar context\
                                           have similar meanings")))
    st.image(root/"SWRG"/"vector_embedding.png")
    st.write_stream(stream_data(text=note("Word embeddings can have varying shapes, for example:")))
    st.write("&emsp;  🔹 GPT-3 175B: 12,288")
    st.write("&emsp;  🔹 Mistral 7B: 4,096")
    st.write("&emsp;  🔹 Gemma-3 1B: 1,152")
    st.write("&emsp;  🔹 GPT-2: 768")


if st.sidebar.button("Section 2.2 - Tokenizing text", type="tertiary"):
    st.session_state.show_section_2_2 = True

if st.session_state.show_section_2_2:
    st.write_stream(stream_data(text=note("In this section we'll talk about splitting input\
                                          text into individual tokens. This step is essential prior\
                                          to obtaining embedding vectors.")))
    # st.session_state.show_section_2_2 = False
    uploaded_file = st.file_uploader("Choose Text File")
    if uploaded_file:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        text_content = stringio.read()
        st.session_state.verdict_content = text_content
        st.code(f"Total number of character: {len(text_content)}")
        st.code(text_content[:99])
    
    st.write_stream(stream_data(text=note("The first question we should ask ourselves is what is the\
                                          best way to split this text?")))
    text_2_parse = st.text_input("Text Input")
    st.code(f"import re\ntext = {text_2_parse}'\nresult = re.split(r'(\s)', text)")
    result = re.split(r"(\s)", text_2_parse)
    st.code(result)
    st.write_stream(stream_data(text=note("Now let's split on whitespaces, commas, and periods\n")))
    st.code("result = re.split(r'([,.]|\s', text)")
    result = re.split(r"([,.]|\s)", text_2_parse)
    st.code(result)
    st.write_stream(stream_data(text=note("Optionally, we can remove whitespaces")))
    st.code("result = [item for item in result if item.strip()]")
    result = [item for item in result if item.strip()]
    st.code(result)
    st.write_stream(stream_data(text=note("Let's finalize our regex command so we can preprocess Verdict.txt")))
    st.code("result = re.split(r'([,.:;?_!\"()\']|--|\s)', text)\nresult = [item.strip() for item in result if item.strip()]")
    
    if st.session_state.verdict_content:
        result = re.split(r"([,.:;?_!\"()\']|--|\s)", text_content)
        result = [item.strip() for item in result if item.strip()]
        st.session_state.verdict_preprocess = result
        st.code(result[:100])
 

if st.sidebar.button("Section 2.3 - Converting tokens into token IDs", type="tertiary"):
    st.session_state.show_section_2_3 = True
      

if st.session_state.show_section_2_3:
    st.write_stream(stream_data(text=note("After breaking up our string, it's now time to convert these tokens into an integer representation.")))
    st.image(root/"SWRG"/"section_2_3_img_1.png")
    st.write_stream(stream_data(text=note("Let's create a list of all unique tokens and sort them alphabetically to determine the vocabulary size.")))
    st.code("all_words = sorted(set(preprocessed))\nvocab_size = len(all_words)")
    preprocessed = st.session_state.verdict_preprocess
    all_words = sorted(set(preprocessed))
    st.code(all_words[:50])
    st.code(len(all_words))
    st.write_stream(stream_data(text=note("Now let's create a simple lookup that'll map token IDs to tokens")))
    st.code("vocab = {token:integer for integer,token in enumerate(all_words)}")
    vocab = {token:integer for integer,token in enumerate(all_words)}
    for i, item in enumerate(vocab.items(), start=50):
        st.code(item)
        if i > 70: break
    st.write_stream(stream_data(text=note("Quick recap of what we just did")))
    st.image(root/"SWRG"/"section_2_3_img_2.png")
    st.write_stream(stream_data(text=note("Now let's create a quick class putting it all together")))
    st.code("""class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab            #1
        self.int_to_str = {i:s for s,i in vocab.items()}        #2

    def encode(self, text):         #3
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):         #4
        text = " ".join([self.int_to_str[i] for i in ids]) 

        text = re.sub(r'\s+([,.?!"()\'])', r'\\1', text)    #5
        return text""")
    st.write_stream(stream_data(text=note("The class above also shows a decode method which will take us from token ids back to tokens")))

    tokenizer = SimpleTokenizerV1(vocab)
    text_2_parse = st.text_input("Text Input (example: it's the last he painted you know?)")
    ids = tokenizer.encode(text_2_parse)
    st.write_stream(stream_data(text=note("Token IDs from above text")))
    st.code(ids)
    st.write_stream(stream_data(text=note("Now let's decode")))
    st.code(tokenizer.decode(ids))

if st.sidebar.button("Section 2.4 - Adding special context tokens", type="tertiary"):
    st.session_state.show_section_2_4 = True

if st.session_state.show_section_2_4:
    preprocessed = st.session_state.verdict_preprocess
    st.write_stream(stream_data(text=note("We need to modify the tokenizer to handle unknown words. In this next section we'll update the tokenizer to handle unknowns and when the end of text has occured")))
    st.image(root/"SWRG"/"section_2_4_img_1.png")
    st.write_stream(stream_data(text=note("We'll first modify tokenizer to use an <|unk|> token if it encounters a word not currently in the vocabulary.\
                                          Then we'll also add a token symbolizes unrelated texts <|endoftext|>")))
    st.code("""all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}""")
    all_tokens = sorted(list(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:integer for integer,token in enumerate(all_tokens)}
    st.write_stream(stream_data(text=note("Previous vocab size: 1130")))
    st.code(len(vocab))
    text_2_parse_1 = st.text_input("Text 1")
    text_2_parse_2 = st.text_input("Text 2")
    complete_text = "<|endoftext|>".join((text_2_parse_1, text_2_parse_2))
    st.code(complete_text)
    tokenizer = SimpleTokenizerV2(vocab)
    st.write_stream(stream_data(text=note("Encoded Token IDs")))
    st.code(tokenizer.encode(complete_text))
    st.write_stream(stream_data(text=note("Decoded Token IDs")))
    st.code(tokenizer.decode(tokenizer.encode(complete_text)))

# if st.sidebar.button("Bonus: Real World Example", type="tertiary"):
#     st.session_state.show_section_real_world = True

# if st.session_state.show_section_real_world:

if __name__=="__main__":
    my_notes = "hello world"
    note(my_notes)