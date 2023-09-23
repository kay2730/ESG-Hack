from collections import OrderedDict
from transformers import MPNetPreTrainedModel, MPNetModel, AutoTokenizer
import torch
import numpy as np
import openai


# Further Improvements 
# Link news feeds on the company's sustainability practices?
# Unstructured data ingestion?
# Could use charts to display some metrics?
# Performance is extremely slow. Use quantised language models locally instead of API?

esg_scorecard = ''

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Definition of ESGify class because of custom,sentence-transformers like, mean pooling function and classifier head
class ESGify(MPNetPreTrainedModel):
    """Model for Classification ESG risks from text."""

    def __init__(self,config): #tuning only the head
        """
        """
        super().__init__(config)
        # Instantiate Parts of model
        self.mpnet = MPNetModel(config,add_pooling_layer=False)
        self.id2label =  config.id2label
        self.label2id =  config.label2id
        self.classifier = torch.nn.Sequential(OrderedDict([('norm',torch.nn.BatchNorm1d(768)),
                                                ('linear',torch.nn.Linear(768,512)),
                                                ('act',torch.nn.ReLU()),
                                                ('batch_n',torch.nn.BatchNorm1d(512)),
                                                ('drop_class', torch.nn.Dropout(0.2)),
                                                ('class_l',torch.nn.Linear(512 ,47))]))


    def forward(self, input_ids, attention_mask):


         # Feed input to mpnet model
        outputs = self.mpnet(input_ids=input_ids,
                             attention_mask=attention_mask)
         
        # mean pooling dataset and eed input to classifier to compute logits
        logits = self.classifier( mean_pooling(outputs['last_hidden_state'],attention_mask))
         
        # apply sigmoid
        logits  = 1.0 / (1.0 + torch.exp(-logits))
        return logits

def esg(company_name):
    # Generate Sample ESG
    openai.api_key = "cUpikAW1hpQpO5jplJiRvIcmJtA7NntM"
    openai.api_base = "https://api.deepinfra.com/v1/openai"

    chat_completion = openai.ChatCompletion.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        messages=[{"role": "user", "content": f"Generate a sample ESG report for the organisation, {company_name}"}],
    )

    texts = [chat_completion.choices[0].message.content]

    model = ESGify.from_pretrained('ai-lab/ESGify')
    tokenizer = AutoTokenizer.from_pretrained('ai-lab/ESGify')
    to_model = tokenizer.batch_encode_plus(
                    texts,
                    add_special_tokens=True,
                    max_length=512,
                    return_token_type_ids=False,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                    )
    results = model(**to_model)

    
    esg_scores = {}
    for i in torch.topk(results, k=25).indices.tolist()[0]:
        esg_scores[model.id2label[i]] = np.round(results.flatten()[i].item(), 3)
        
    return esg_scores


def chat(esg_scores, query):
    esg_scores = str(esg_scores)
    
    openai.api_key = "cUpikAW1hpQpO5jplJiRvIcmJtA7NntM"
    openai.api_base = "https://api.deepinfra.com/v1/openai"

    chat_completion = openai.ChatCompletion.create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": f"This is the ESG score for a company: {esg_scores}. Based on the scores, answer this question {query}"}],
        )
     
    return chat_completion.choices[0].message.content

def improvement_suggestion(esg_scores):
    esg_scores = str(esg_scores)
    
    openai.api_key = "cUpikAW1hpQpO5jplJiRvIcmJtA7NntM"
    openai.api_base = "https://api.deepinfra.com/v1/openai"

    chat_completion = openai.ChatCompletion.create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": f"This is the ESG score for a company: {esg_scores}. Suggest 5 ways to the company to improve the score. Answer in very short bullet points. The answer should be brief, concise and to the point."}],
        )
     
    return chat_completion.choices[0].message.content


