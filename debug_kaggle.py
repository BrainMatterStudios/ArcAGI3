import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
try:
    from kaggle.rest import ApiException
except ImportError:
    pass
import json

def main():
    api = KaggleApi()
    api.authenticate()
    try:
        res = api.competition_submit_code('submission.parquet', 'Auto submit test', 'arc-prize-2026-arc-agi-3', 'ahmedmobasher86/arc-agi-3-duck-patched', 4)
        print("Success:", res)
    except Exception as e:
        print("Exception:", e)
        if hasattr(e, 'response'):
            print("Response:", e.response.text)
            print("Body:", e.body)
            try:
                print("Parsed body:", json.loads(e.body))
            except:
                pass
        if hasattr(e, 'reason'):
            print("Reason:", e.reason)

if __name__ == "__main__":
    main()
