from typing import List, Optional
import abc


class Retriever():
    @abc.abstractmethod
    def retrieve(
        self,
        query: str,
        topk: int=1000,
    ) -> List[dict]:
        """
        Rretrieve tweets from a indexed corpus with respect to a query.
        Parameters
        ----------
        query : str
            The query.
        topk : int
            topk relevant tweets to return.
        Returns
        -------
        List[dict]
            retrieve a list of tweet dict.
        """
        pass
