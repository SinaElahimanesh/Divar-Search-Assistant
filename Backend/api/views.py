from django.shortcuts import render

# Create your views here.

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import json
from sentence_transformers import SentenceTransformer
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

from core import main

class Chat(APIView):
    state = "GREETING"
    FINAL_STATE = "END2"
    buttons = []
    previous_questions_embeddings = []
    model = SentenceTransformer('HooshvareLab/bert-base-parsbert-uncased')


    model_size = "small"
    model_name = f"persiannlp/mt5-{model_size}-parsinlu-squad-reading-comprehension"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    comprehension_model = MT5ForConditionalGeneration.from_pretrained(model_name)
  
    def post(self, request):
        message = json.loads(request.body.decode('utf-8'))["message"]
        if message == "restart":
            Chat.state = "GREETING"
        res, Chat.state, Chat.buttons, Chat.previous_questions_embeddings = main.information_retrieval_module(Chat.state, message, Chat.previous_questions_embeddings, Chat.model, Chat.comprehension_model, Chat.tokenizer)
        if Chat.state != Chat.FINAL_STATE:
            return Response({"status": "success", "response": res, "buttons": Chat.buttons},
                            status=status.HTTP_200_OK)
        return Response({"status": "end", "response":  "امیدوارم تونسته باشم کمکت کنم.", "buttons": Chat.buttons}, status=status.HTTP_200_OK)
