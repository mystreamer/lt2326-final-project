from external.prepare_data import main 

if __name__ == "__main__":
    main(
        input_dir="./data/qt-30/train",
        output_dir="./data/normalised",
        integrate_gold_data=True, # Only in the case of evlauation
        nodeset_id=17918,
        nodeset_blacklist=None,
        nodeset_whitelist=None,
        s_node_type="RA",
        s_node_text="NONE",
        ya_node_text="NONE",
    )
    print("Preparing data successfully completed!")