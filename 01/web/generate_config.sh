#!/bin/bash
# Usage: ./generate_config.sh bm25|vector|hybrid

CONFIG_TYPE=$1

case $CONFIG_TYPE in
    bm25)
        export CONFIG_NAME="BM25"
        export ES_INDEX="datatalks-faqs"
        export SEARCH_TYPE="bm25"
        export USE_VECTOR="false"
        export BOOST_QUESTION="20.0"
        export BOOST_TEXT="1.0"
        ;;
    vector)
        export CONFIG_NAME="Vector"
        export ES_INDEX="datatalks-faqs"
        export SEARCH_TYPE="vector"
        export USE_VECTOR="true"
        ;;
    hybrid)
        export CONFIG_NAME="Hybrid"
        export ES_INDEX="datatalks-faqs"
        export SEARCH_TYPE="hybrid"
        export USE_VECTOR="true"
        export BOOST_QUESTION="20.0"
        export BOOST_TEXT="1.0"
        ;;
    *)
        echo "Usage: $0 {bm25|vector|hybrid}"
        exit 1
        ;;
esac

# Substitute env vars in template
envsubst < experiments/configs/template.json > experiments/configs/${CONFIG_TYPE}.json
echo "Generated experiments/configs/${CONFIG_TYPE}.json"
