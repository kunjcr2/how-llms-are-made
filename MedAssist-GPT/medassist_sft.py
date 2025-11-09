"""
Quick Script: Load Your Pretrained Checkpoint and Fine-tune
============================================================
This script loads your pretrained model checkpoint and fine-tunes it
Handles both local checkpoints and HuggingFace models
"""

import torch
from pathlib import Path
from medassist_pretrain import *  # Import everything from the SFT script

# ============================================================================
# CONFIGURATION - EDIT THESE!
# ============================================================================

# Your pretrained checkpoint (choose one)
CHECKPOINT_OPTIONS = {
    # Option 1: Local checkpoint from pretraining
    "local": "./checkpoints/checkpoint_step_15000.pt",
    
    # Option 2: HuggingFace repo (after you upload)
    "huggingface": "YOUR_USERNAME/MedAssist-GPT-500M",
    
    # Option 3: Best checkpoint (auto-find)
    "auto_best": "./checkpoints",
}

# Which option to use?
USE_CHECKPOINT = "auto_best"  # Change to "huggingface" or "auto_best"

# ============================================================================
# CHECKPOINT LOADING UTILITIES
# ============================================================================

def find_best_checkpoint(checkpoint_dir: str) -> str:
    """
    Automatically find the best checkpoint in a directory
    Returns the checkpoint with the highest step number
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
    best_checkpoint = checkpoints[-1]
    
    print(f"📂 Found {len(checkpoints)} checkpoints")
    print(f"✅ Using latest: {best_checkpoint.name}")
    
    return str(best_checkpoint)


def load_local_checkpoint(checkpoint_path: str, config: Dict) -> nn.Module:
    """
    Load model from local checkpoint file
    """
    print(f"🔧 Loading checkpoint from {checkpoint_path}...")
    
    # Initialize model
    model = MedAssistGPT(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📊 Checkpoint info:")
        print(f"  - Step: {checkpoint.get('step', 'unknown')}")
        print(f"  - Loss: {checkpoint.get('loss', 'unknown'):.4f}")
    else:
        # Just weights, no metadata
        model.load_state_dict(checkpoint)
    
    print("✅ Model loaded successfully!")
    return model


def load_from_huggingface(repo_id: str, config: Dict) -> nn.Module:
    """
    Load model from HuggingFace Hub
    """
    print(f"🔧 Loading model from HuggingFace: {repo_id}...")
    
    from huggingface_hub import hf_hub_download
    
    try:
        # Download model file
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="pytorch_model.bin"
        )
        
        # Initialize and load
        model = MedAssistGPT(config)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        print("✅ Model loaded from HuggingFace!")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load from HuggingFace: {e}")
        print("💡 Make sure you've uploaded your model to HuggingFace!")
        raise


def load_model_smart(config: Dict) -> nn.Module:
    """
    Smart loader - tries multiple methods
    """
    global USE_CHECKPOINT, CHECKPOINT_OPTIONS
    
    if USE_CHECKPOINT == "local":
        checkpoint_path = CHECKPOINT_OPTIONS["local"]
        if not Path(checkpoint_path).exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            print("🔍 Trying to auto-find best checkpoint...")
            USE_CHECKPOINT = "auto_best"
        else:
            return load_local_checkpoint(checkpoint_path, config)
    
    if USE_CHECKPOINT == "auto_best":
        checkpoint_dir = CHECKPOINT_OPTIONS["auto_best"]
        checkpoint_path = find_best_checkpoint(checkpoint_dir)
        return load_local_checkpoint(checkpoint_path, config)
    
    if USE_CHECKPOINT == "huggingface":
        repo_id = CHECKPOINT_OPTIONS["huggingface"]
        return load_from_huggingface(repo_id, config)
    
    raise ValueError(f"Unknown checkpoint option: {USE_CHECKPOINT}")


# ============================================================================
# MAIN FINE-TUNING SCRIPT
# ============================================================================

def main_with_checkpoint_loading():
    """
    Main function that loads checkpoint and fine-tunes
    """
    
    print("="*80)
    print("🚀 MEDASSIST-GPT: LOAD PRETRAINED & FINE-TUNE")
    print("="*80)
    
    # Set seeds
    torch.manual_seed(TRAINING_CONFIG["seed"])
    np.random.seed(TRAINING_CONFIG["seed"])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # Create save directory
    save_dir = Path("./sft_checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # STEP 1: LOAD PRETRAINED MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: Loading Pretrained Model")
    print("="*80)
    
    model = load_model_smart(MODEL_CONFIG)
    model = model.to(device)
    
    print(f"✅ Model has {model.count_parameters():,} parameters")
    
    # ========================================================================
    # STEP 2: ADD LORA ADAPTERS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Adding LoRA Adapters")
    print("="*80)
    
    model = setup_lora(model, LORA_CONFIG)
    
    # ========================================================================
    # STEP 3: LOAD DATA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Loading Fine-tuning Data")
    print("="*80)
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("p50k_base")
    
    # Load medical QA data
    train_examples, val_examples = load_medical_qa_data(DATA_CONFIG)
    
    # Create dataloaders
    train_loader = create_sft_dataloader(
        train_examples,
        tokenizer,
        batch_size=TRAINING_CONFIG["batch_size"],
        max_length=TRAINING_CONFIG["max_seq_length"],
        shuffle=True,
        num_workers=TRAINING_CONFIG["num_workers"]
    )
    
    val_loader = create_sft_dataloader(
        val_examples,
        tokenizer,
        batch_size=TRAINING_CONFIG["batch_size"],
        max_length=TRAINING_CONFIG["max_seq_length"],
        shuffle=False,
        num_workers=TRAINING_CONFIG["num_workers"]
    )
    
    # ========================================================================
    # STEP 4: SETUP TRAINING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Setting Up Optimizer & Scheduler")
    print("="*80)
    
    # Optimizer (only train LoRA parameters)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"]
    )
    
    # Learning rate scheduler
    from transformers import get_linear_schedule_with_warmup
    
    total_steps = (
        len(train_loader) * TRAINING_CONFIG["num_epochs"] 
        // TRAINING_CONFIG["gradient_accumulation_steps"]
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=TRAINING_CONFIG["warmup_steps"],
        num_training_steps=total_steps
    )
    
    print(f"📊 Total training steps: {total_steps:,}")
    print(f"📊 Warmup steps: {TRAINING_CONFIG['warmup_steps']}")
    
    # ========================================================================
    # STEP 5: INITIALIZE TRACKING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Initializing Experiment Tracking")
    print("="*80)
    
    # Initialize wandb
    wandb.init(
        project=WANDB_CONFIG["project"],
        entity=WANDB_CONFIG["entity"],
        name=WANDB_CONFIG["name"],
        config={
            **MODEL_CONFIG,
            **LORA_CONFIG,
            **TRAINING_CONFIG,
            "checkpoint_used": USE_CHECKPOINT,
        }
    )
    
    # HuggingFace setup
    if HF_CONFIG.get("upload_checkpoints"):
        try:
            from huggingface_hub import login, create_repo
            login()
            create_repo(HF_CONFIG["sft_repo_id"], repo_type="model", exist_ok=True)
            print(f"✅ HuggingFace repo ready: {HF_CONFIG['sft_repo_id']}")
        except Exception as e:
            print(f"⚠️  HuggingFace setup skipped: {e}")
    
    # ========================================================================
    # STEP 6: FINE-TUNE!
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: Starting Fine-Tuning")
    print("="*80)
    
    trained_model = train_sft(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=TRAINING_CONFIG,
        tokenizer=tokenizer,
        save_dir=save_dir
    )
    
    # ========================================================================
    # STEP 7: EVALUATE & SAVE
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: Final Evaluation & Model Merging")
    print("="*80)
    
    # Calculate final metrics
    metrics = calculate_metrics(trained_model, val_loader, device)
    
    # Log final metrics
    wandb.log({
        "final_val_loss": metrics["loss"],
        "final_perplexity": metrics["perplexity"],
        "final_accuracy": metrics["accuracy"],
    })
    
    # Merge LoRA weights for easy deployment
    print("\n🔧 Merging LoRA weights with base model...")
    merged_model = merge_and_save_model(
        lora_model=trained_model,
        save_path="./merged_model",
        push_to_hub=HF_CONFIG.get("upload_checkpoints", False),
        repo_id=HF_CONFIG.get("sft_repo_id")
    )
    
    # ========================================================================
    # STEP 8: TEST GENERATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 8: Testing Final Model")
    print("="*80)
    
    test_questions = [
        "What are the main symptoms of Type 2 diabetes?",
        "How does the human immune system fight infections?",
        "Explain the difference between arteries and veins.",
        "What is hypertension and how is it treated?",
    ]
    
    print("\n📝 Sample Generations from Fine-Tuned Model:\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"Question {i}: {question}")
        print(f"{'='*70}")
        
        response = generate_sample(
            model=merged_model,
            tokenizer=tokenizer,
            prompt=question,
            max_new_tokens=150,
            device=device
        )
        
        print(f"Answer:\n{response}\n")
    
    # Finish
    wandb.finish()
    
    print("\n" + "="*80)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*80)
    print(f"\n📂 LoRA adapters saved in: {save_dir}")
    print(f"📂 Merged model saved in: ./merged_model")
    print(f"\n🎯 Final Metrics:")
    print(f"  - Validation Loss: {metrics['loss']:.4f}")
    print(f"  - Perplexity: {metrics['perplexity']:.2f}")
    print(f"  - Token Accuracy: {metrics['accuracy']:.4f}")
    print(f"\n🚀 Next: Use this model for DPO training!")
    print("="*80)


# ============================================================================
# QUICK TEST: VERIFY CHECKPOINT LOADS
# ============================================================================

def test_checkpoint_loading():
    """
    Quick test to verify your checkpoint loads correctly
    """
    print("🧪 Testing Checkpoint Loading...\n")
    
    try:
        model = load_model_smart(MODEL_CONFIG)
        print(f"✅ Checkpoint loaded successfully!")
        print(f"📊 Model has {model.count_parameters():,} parameters")
        
        # Quick forward pass test
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        dummy_input = torch.randint(0, 50257, (1, 128), device=device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Forward pass successful!")
        print(f"📊 Output shape: {output.shape}")
        print("\n🎉 Everything looks good! Ready to fine-tune!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("\n💡 Troubleshooting:")
        print("  1. Check that CHECKPOINT_OPTIONS paths are correct")
        print("  2. Verify checkpoint file exists")
        print("  3. Make sure MODEL_CONFIG matches your pretrained model")
        return False


if __name__ == "__main__":
    # Quick test mode: just verify checkpoint loads
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_checkpoint_loading()
    else:
        # Full fine-tuning
        main_with_checkpoint_loading()