import argparse
import time
import torch
from datetime import timedelta
from transformers import BertTokenizer

from utils import transform
from models import ImageCaptionModel
from evaluation import generate_caption



def main():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--embedding_dim", "-ed", type=int, default=512, help="Embedding dimension (embedding_dim must be a divisor of 7*7*2048)")
    parser.add_argument("--attention_dim", "-ad", type=int, default=256, help="Attention dim")
    parser.add_argument("--tokenizer", "-t", type=str, default="bert-base-uncased", help="Bert tokenizer")
    parser.add_argument("--max_seq_len", "-msl", type=int, default=128, help="Maximum sequence length for caption generation")
    parser.add_argument("--num_layers", "-nl", type=int, default=8, help="Number of layers in the transformer decoder")
    parser.add_argument("--num_heads", "-nh", type=int, default=8, help="Number of heads in multi-head attention")
    parser.add_argument("--dropout", "-dr", type=float, default=0.1, help="Dropout probability")
    # Training parameters
    parser.add_argument("--model_path", "-md", type=str, default="./pretrained/model_image_captioning_eff_transfomer.pt", help="Path to save model")
    parser.add_argument("--device", "-d", type=str, default="cuda:0", help="Device to use {cpu, cuda:0, cuda:1,...}")
    parser.add_argument("--beam_size", "-b", type=int, default=3, help="Beam size for beam search")
    args = parser.parse_args()

    # Load model and tokenizer
    device = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model_configs = {
        "embedding_dim": args.embedding_dim,
        "attention_dim": args.attention_dim,
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": args.max_seq_len,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
    }
    # Load model ImageCaptionModel
    start_time = time.time()
    model = ImageCaptionModel(**model_configs)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    time_load_model = str(timedelta(seconds=int(time.time() - start_time)))
    print(f"Done load model on the {device} device in {time_load_model}")

    # Generate captions
    while True:
        image_path = input("Enter image path (or q to exit): ")
        if image_path == "q":
            break
        st = time.time()
        cap = generate_caption(
            model=model,
            image_path=image_path,
            transform=transform,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            beam_size=args.beam_size,
            device=device,
            print_process=False
        )
        end = time.time()
        print("--- Caption: {}".format(cap))
        print(f"--- Time: {end-st} (s)")
        

if __name__ == "__main__":
    main()