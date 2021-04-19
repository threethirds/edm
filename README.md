# EDM

# dockers

    cd edm/

    docker build -t eu.gcr.io/ai4eu-33/edm-agent -f ai4eu_agent/Dockerfile .
    docker build -t eu.gcr.io/ai4eu-33/edm-env -f ai4eu_env/Dockerfile .

    docker push eu.grc.io/ai4eu-33/edm-agent
    docker push eu.grc.io/ai4eu-33/edm-env
