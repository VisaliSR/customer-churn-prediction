import pandas as pd
from src.custom_transformers import BinaryMapper
def test_binary_mapper():

    df=pd.DataFrame({
        "gender":["Male","Female","Male"]
    })
    mappings={
        "gender":{"Male":1,"Female":0}
    }
    mapper=BinaryMapper(mappings)
    result=mapper.transform(df)
    assert result["gender"].tolist()==[1,0,1]
