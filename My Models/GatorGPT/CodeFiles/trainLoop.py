from trainFunc import *

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize tracking lists
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n🔁 Starting epoch {epoch+1}")
        model.train()
        epoch_improved = False

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                # Check for validation improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epoch_improved = True

        # Print sample after epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

        # If no improvement this epoch, skip rest
        if not epoch_improved:
            print(f"⚠️ No val loss improvement in epoch {epoch+1}, skipping to next.")
            continue

    return train_losses, val_losses, track_tokens_seen