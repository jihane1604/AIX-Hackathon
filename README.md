# AIX-Hackathon
Repository for our AIX hackathon project :)
## Team members
- Mustafa Mohamed
- Mohammad Thaher
- Ruqaya Ali
- Djihane Mahraz

# Step 1 scaffold ready.

Run:
  conda env create -f environment.yml
  conda activate aix-hackathon-env
  pip install -r AIX-HACKATHON/requirements.txt
  python AIX-HACKATHON/src/step1_build_manifest.py

Outputs:
  data/interim/manifest.json          # registry of all inputs
  data/interim/<doc_id>.txt           # normalized extracts per file

- This part of the code does not need gpu acceleration (trivial tasks)
- Manifest is created solely based on metadata, they will be updated later on
- It's scalable because we can add whatever regulatory corpus documents (like one uploaded by the user) to data/raw and it run the file again
- But right now it cant automatically add regulators to the folder which i think will cause a problem later, theres some gpt code for getting a user post with the information on the regulator (name, jursidiction, domains) then it automatically makes it but its not working, someone (mr software engineer) should look at it pls -- command to run `uvicorn src.server:app --reload --host 0.0.0.0 --port 8000`

# Step 2 enrichment and rulepack generation.
- Modules in step a: aliases.py, chunker.py, enrich.py, rulepack_from_docs.py, step2_enrich_and_build_rulepacks.py
- Uncomment line 92 in `model_rulepack.py` for gpu (does not take too long on cpu but just in case)
- Updates the manifest to add more info to each doc provided 
- For step 2b i need someone (mr business man) to give me the full list of domains
- Modules in step b: model_rulepack.py, rulepack_merge.py, step2b_generate_rulepacks_model.py