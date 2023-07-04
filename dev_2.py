def main():
    from peft import TaskType, GlobalMemLoraConfig
    from peft.tuners.discrete_kv_tuning import Global_Memory_KV_Lora
    import torch

    peft_config = GlobalMemLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        codebook_input_channel=8,
        codebook_num=1,
        kv_pairs_num=100,
    )
    net = Global_Memory_KV_Lora(32, 4, peft_config).to("cuda")

    x = torch.randn([32, 1000, 32]) / 8
    y = 1 * x
    x = x.to("cuda")
    y = y.to("cuda")
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(1000):
        net.train()
        output = net(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        print("loss\t", loss.item())
        opt.step()


if __name__ == "__main__":
    main()
