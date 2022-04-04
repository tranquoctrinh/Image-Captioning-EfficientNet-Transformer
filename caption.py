import argparse
from utils import configs, transform
from evaluation import load_model_tokenizer, generate_caption, preprocess_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="./images/test.jpg")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--beam_size", type=int, default=3)
    args = parser.parse_args()

    model, tokenizer, device = load_model_tokenizer(configs)

    st = time.time()
    cap = generate_caption(
        model=model,
        image=preprocess_image(args.image_path, transform),
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