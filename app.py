import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer
from fugashi import Tagger
import torch

st.title('感情分析')
st.caption('感情分析するやつです')

text = st.text_input('分析する文章')

if not text.strip():
    st.warning("テキストを入力してください。")
else:
    # 日本語から英語に翻訳
    translator = Translator()
    try:
        text_to_en = translator.translate(text, src='ja', dest='en').text
    except Exception as e:
        st.error("翻訳中にエラーが発生しました: " + str(e))
        text_to_en = ""

    if text_to_en:
        try:
            # VADERの分析
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text_to_en)

            st.text('＜VADER（ルールベースの感情分析ライブラリ）での評価＞')
            st.text(f'Positive Score: {scores["pos"]}')
            st.text(f'Negative Score: {scores["neg"]}')
            st.text(f'Neutral Score: {scores["neu"]}')
        except Exception as e:
            st.error("VADERによる分析中にエラーが発生しました: " + str(e))

    try:
        st.text('＜BERT（事前学習の機械学習モデル）での評価＞')

        # BERTの分析
        class DocmentsAnalysis:
            def __init__(self, model='christian-phu/bert-finetuned-japanese-sentiment',
                         tokenizer='tohoku-nlp/bert-base-japanese-v2'):
                self.tokenizer = BertJapaneseTokenizer.from_pretrained(
                    tokenizer,
                    do_subword_tokenize=True
                )

                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model,
                    output_attentions=True
                )

                self.pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, top_k=None)

        da = DocmentsAnalysis()
        bert_result = da.pipe(text)

        scores = {item['label']: item['score'] for item in bert_result[0]}
        st.text(f'Positive Score: {scores.get("positive", "N/A")}')
        st.text(f'Negative Score: {scores.get("negative", "N/A")}')
        st.text(f'Neutral Score: {scores.get("neutral", "N/A")}')
    except Exception as e:
        st.error("BERTによる分析中にエラーが発生しました: " + str(e))
