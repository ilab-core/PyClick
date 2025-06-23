#
# Copyright (C) 2025  Afra Arslan, Hacer Turgut
#
# Full copyright notice can be found in LICENSE.
#
import math
from enum import Enum
from pyclick.click_models.ClickModel import ClickModel
from pyclick.click_models.Inference import EMInference
from pyclick.click_models.Param import ParamEM, ParamMLE, ParamStatic
from pyclick.click_models.ParamContainer import (QueryDocumentParamContainer,
                                                 SingleParamContainer)


class iCCDBN(ClickModel):
    """
    """

    param_names = Enum('iCCDBNParamNames', 'attr sat_click sat_pur pur cont exam car par')
    """
    The names of the iCCDBN parameters.

    :attr: the attractiveness parameter.
    Determines whether a user clicks on a search results after examining it.
    :sat_click: the statisfactoriness parameter.
    Determines whether a user is satisfied with a search result (and abandons the corresponding search session)
    after clicking and reading the result, without purchase event.
    :sat_pur: the statisfactoriness parameter.
    Determines whether a user is satisfied with a search result (and abandons the corresponding search session)
    after clicking and reading the result, with purchase event.
    :pur: the purchase parameter.
    Determines whether a user purchases search results after clicking the current one.
    :cont: the continuation parameter.
    Determines whether a user continues examining search results after examining the current one.

    :exam: the examination probability.
    Not defined explicitly in the iCCDBN model, but needs to be calculated during inference.
    Determines whether a user examines a particular search result.
    :car: the probability of click on or after rank $r$ given examination at rank $r$.
    Not defined explicitly in the iCCDBN model, but needs to be calculated during inference.
    Determines whether a user clicks on the current result or any result below the current one.
    :par: the probability of purchase on or after rank $r$ given examination at rank $r$.
    Not defined explicitly in the iCCDBN model, but needs to be calculated during inference.
    Determines whether a user purchase on the current result or any result below the current one.
    """

    def __init__(self, inference=EMInference(), cont_static=0.95):
        self.params = {self.param_names.attr: QueryDocumentParamContainer(iCCDBNAttrEM),  # self._container[query][search_result]
                       self.param_names.sat_click: QueryDocumentParamContainer(iCCDBNSatClickEM),
                       self.param_names.sat_pur: QueryDocumentParamContainer(iCCDBNSatPurEM),
                       self.param_names.pur: QueryDocumentParamContainer(iCCDBNPurMLE),
                       self.param_names.cont: SingleParamContainer(iCCDBNContEM, static=cont_static)}

        self._inference = inference

    def get_session_params(self, search_session):
        session_params = super(iCCDBN, self).get_session_params(search_session)

        session_exam = self._get_session_exam(search_session, session_params)
        session_clickafterrank = self._get_session_clickafterrank(search_session, session_params)

        for rank, session_param in enumerate(session_params):
            session_param[self.param_names.exam] = ParamStatic(session_exam[rank])
            session_param[self.param_names.car] = ParamStatic(session_clickafterrank[rank])

        return session_params

    def get_full_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        click_probs = []

        for _, session_param in enumerate(session_params):
            attr = session_param[self.param_names.attr].value()
            exam = session_param[self.param_names.exam].value()

            click_probs.append(attr * exam)

        return click_probs

    def get_full_purchase_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        purchase_probs = []

        for _, session_param in enumerate(session_params):
            attr = session_param[self.param_names.attr].value()
            pur = session_param[self.param_names.pur].value()
            exam = session_param[self.param_names.exam].value()

            purchase_probs.append(attr * pur * exam)

        return purchase_probs

    def get_conditional_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        return self._get_tail_probs(search_session, 0, session_params)[0]

    def get_conditional_purchase_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        return self._get_tail_probs(search_session, 0, session_params)[1]

    def predict_relevance(self, query, search_result):
        attr = self.params[self.param_names.attr].get(query, search_result).value()
        sat_click = self.params[self.param_names.sat_click].get(query, search_result).value()
        sat_pur = self.params[self.param_names.sat_pur].get(query, search_result).value()
        pur = self.params[self.param_names.pur].get(query, search_result).value()
        return attr * (sat_click * (1 - pur) + sat_pur * pur)

    def _get_session_exam(self, search_session, session_params):
        """
        Calculates the examination probability P(E_{r+1} = 1) for each search result in a given search session.

        :param search_session: The observed search session.
        :param session_params: The current values of parameters for a given search session.

        :returns: The list of examination probabilities for a given search session.
        """
        session_exam = [1]

        for rank, session_param in enumerate(session_params):
            attr = session_param[self.param_names.attr].value()
            sat_click = session_param[self.param_names.sat_click].value()
            sat_pur = session_param[self.param_names.sat_pur].value()
            cont = session_param[self.param_names.cont].value()
            pur = session_param[self.param_names.pur].value()
            exam = session_exam[rank]

            exam *= cont * ((1 - sat_pur) * pur * attr + (1 - sat_click) * (1 - pur) * attr + (1 - attr))
            session_exam.append(exam)

        return session_exam

    @classmethod
    def _get_tail_probs(cls, search_session, start_rank, session_params):
        """
        Calculate P(C_r | C_{r-1}, ..., C_l, E_l = 1),
        P(T_r | C_{r-1}, ..., C_l, E_l = 1), P(E_r = 1 | C_{r-1}, ..., C_l, E_l = 1)
        for each r in [l, n) where l is start_rank.
        """
        exam = 1.0
        click_probs = []
        purchase_probs = []
        exam_probs = [exam]
        for rank, result in enumerate(search_session.web_results[start_rank:]):
            attr = session_params[rank][cls.param_names.attr].value()
            sat_click = session_params[rank][cls.param_names.sat_click].value()
            sat_pur = session_params[rank][cls.param_names.sat_pur].value()
            cont = session_params[rank][cls.param_names.cont].value()
            pur = session_params[rank][cls.param_names.pur].value()
            if result.click:
                if result.purchase:
                    not_sat = (1 - sat_pur)
                else:
                    not_sat = (1 - sat_click)
                click_prob = attr * exam
                purchase_prob = attr * exam * pur
                exam = cont * not_sat
            else:
                if not result.purchase:
                    click_prob = 1 - attr * exam
                    purchase_prob = 1 - attr * exam * pur
                    exam *= cont * (1 - attr) / click_prob
            click_probs.append(click_prob)
            purchase_probs.append(purchase_prob)
            exam_probs.append(exam)
        return click_probs, purchase_probs, exam_probs

    @classmethod
    def _get_continuation_factor(cls, search_session, rank, session_params):
        """Calculate P(E_r = x, S_r = y, E_{r+1} = z, C, T) up to a constant."""
        click = search_session.web_results[rank].click
        purchase = search_session.web_results[rank].purchase
        attr = session_params[rank][cls.param_names.attr].value()
        sat_click = session_params[rank][cls.param_names.sat_click].value()
        sat_pur = session_params[rank][cls.param_names.sat_pur].value()
        pur = session_params[rank][cls.param_names.pur].value()
        cont = session_params[rank][cls.param_names.cont].value()

        def factor(x, y, z):
            log_prob = 0.0
            if not click:
                if not purchase:
                    if y:
                        # no click -> no satisfaction
                        return 0.0
                    if x:
                        log_prob += math.log(1 - attr)
                        log_prob += math.log(cont if z else (1 - cont))
                    elif z:
                        # no examination at r -> no examination at r+1
                        return 0.0
                else:
                    return 0.0
            else:
                if not x:
                    # no examination at r -> no click
                    return 0.0
                log_prob += math.log(attr)
                if not purchase:
                    log_prob += math.log(1 - pur)
                    if not y:
                        log_prob += math.log(1 - sat_click)
                        log_prob += math.log(cont if z else (1 - cont))
                    else:
                        if z:
                            # satisfaction at r -> no examination at r+1
                            return 0.0
                        log_prob += math.log(sat_click)
                else:
                    log_prob += math.log(pur)
                    if not y:
                        log_prob += math.log(1 - sat_pur)
                        log_prob += math.log(cont if z else (1 - cont))
                    else:
                        if z:
                            return 0.0
                        log_prob += math.log(sat_pur)
            # Then we compute P(\mathbf{C}_{>r}, \mathbf{T}_{>r} | E_{r+1} = z)
            if not z:
                if search_session.get_last_click_rank() >= rank + 1:
                    # no examination -> no clicks
                    return 0.0
                elif search_session.get_last_purchase_rank() >= rank + 1:
                    return 0.0
            elif rank + 1 < len(search_session.web_results):
                if search_session.get_last_purchase_rank() > search_session.get_last_click_rank():
                    return 0.0
                # P(\mathbf{T}_{>r} | \mathbf{C}_{>r}, E_{r+1} = z) · P(\mathbf{C}_{>r} | E_{r+1} = 1)
                purchase_probs = cls._get_tail_probs(search_session, rank + 1, session_params)[1]
                log_prob += sum(math.log(p) for p in purchase_probs)
            # Finally, we compute P(E_r = x | \mathbf{C}_{<r}, \mathbf{T}_{<r})
            exam = cls._get_tail_probs(search_session, 0, session_params)[2][rank]
            log_prob += math.log(exam if x else (1 - exam))
            return math.exp(log_prob)
        return factor

    @classmethod
    def _get_total_continuation_factor(cls, search_session, rank, session_params):
        """Calculate P(E_r = x, S_r = y, E_{r+1} = z, C) for all x, y, z and sum them up."""
        click = search_session.web_results[rank].click
        purchase = search_session.web_results[rank].purchase
        attr = session_params[rank][cls.param_names.attr].value()
        sat_click = session_params[rank][cls.param_names.sat_click].value()
        sat_pur = session_params[rank][cls.param_names.sat_pur].value()
        pur = session_params[rank][cls.param_names.pur].value()
        cont = session_params[rank][cls.param_names.cont].value()

        log_prob = 0.0
        purchase_prob = sum(
            math.log(p) for p in cls._get_tail_probs(search_session, rank + 1, session_params)[1])

        exam = cls._get_tail_probs(search_session, 0, session_params)[2][rank]
        if not click:
            if not purchase:
                if search_session.get_last_click_rank() < rank + 1:
                    log_prob += math.exp(math.log(1 - exam))
                    log_prob += math.exp(math.log(1 - attr) + math.log(1 - cont) + math.log(exam))
                log_prob += math.exp(math.log(1 - attr) + math.log(cont) + purchase_prob + math.log(exam))
        else:
            if not purchase:
                last_prob = 0.0
                if search_session.get_last_click_rank() < rank + 1:
                    log_prob += math.exp(math.log(attr) + math.log(1 - pur) + math.log(1 - sat_click) + math.log(1 - cont) + math.log(exam))
                    last_prob = math.exp(math.log(attr) + math.log(1 - pur) + math.log(sat_click) + math.log(exam))
                log_prob += math.exp(math.log(attr) + math.log(1 - pur) + math.log(1 - sat_click) + math.log(cont) + purchase_prob + math.log(exam))
                log_prob += last_prob
            else:
                last_prob = 0.0
                if search_session.get_last_click_rank() < rank + 1:
                    log_prob += math.exp(math.log(attr) + math.log(pur) + math.log(1 - sat_pur) + math.log(1 - cont) + math.log(exam))
                    last_prob = math.exp(math.log(attr) + math.log(pur) + math.log(sat_pur) + math.log(exam))
                log_prob += math.exp(math.log(attr) + math.log(pur) + math.log(1 - sat_pur) + math.log(cont) + purchase_prob + math.log(exam))
                log_prob += last_prob
        return log_prob

    def _get_session_clickafterrank(self, search_session, session_params):
        """
        For each search result in a given search session,
        calculates the probability of a click on the current result
        or any result below the current one given examination at the current rank,
        i.e., P(C_{>=r} = 1 | E_r = 1), where r is the rank of the current search result.

        :param search_session: The observed search session.
        :param session_params: The current values of parameters for a given search session.

        :returns: The list of P(C_{>=r} = 1 | E_r = 1) for a given search session.
        """
        session_clickafterrank = [0] * (len(search_session.web_results) + 1)

        for rank in range(len(search_session.web_results) - 1, -1, -1):
            attr = session_params[rank][self.param_names.attr].value()
            cont = session_params[rank][self.param_names.cont].value()
            car = session_clickafterrank[rank + 1]  # X^r+1

            car = attr + (1 - attr) * cont * car
            session_clickafterrank[rank] = car

        return session_clickafterrank

    def _get_session_purchaseafterrank(self, search_session, session_params):
        """
        For each search result in a given search session,
        calculates the probability of a purchase on the current result
        or any result below the current one given examination at the current rank,
        i.e., P(T_{>=r} = 1 | C_{>=r}, E_r = 1), where r is the rank of the current search result.

        :param search_session: The observed search session.
        :param session_params: The current values of parameters for a given search session.

        :returns: The list of P(T_{>=r} = 1 | C_{>=r}, E_r = 1) for a given search session.
        """
        session_purchaseafterrank = [0.0] * (len(search_session.web_results) + 1)

        for rank in range(len(search_session.web_results) - 1, -1, -1):
            click = search_session.web_results[rank].click
            pur = session_params[rank][self.param_names.pur].value()
            car = session_params[rank][self.param_names.car].value()
            cont = session_params[rank][self.param_names.cont].value()
            sat_click = session_params[rank][self.param_names.sat_click].value()
            sat_pur = session_params[rank][self.param_names.sat_pur].value()

            par = session_purchaseafterrank[rank + 1]  # Y^r+1
            if click:
                par = pur + par * (1 - pur) * car * cont * ((1 - sat_click) * (1 - pur) + (1 - sat_pur) * pur)

            session_purchaseafterrank[rank] = par

        return session_purchaseafterrank


class iCCDBNAttrEM(ParamEM):
    """
    The attractiveness parameter of the iCCDBN model.
    The value of the parameter is inferred using the EM algorithm.
    """

    def update(self, search_session, rank, session_params):  # TO-DO: MLE yapilabilir mi?
        if search_session.web_results[rank].click:  # cu
            self._numerator += 1
        elif rank >= search_session.get_last_click_rank():  # (1 - cu)(1 - c>r)
            attr = session_params[rank][iCCDBN.param_names.attr].value()
            exam = session_params[rank][iCCDBN.param_names.exam].value()
            car = session_params[rank][iCCDBN.param_names.car].value()

            num = (1 - exam) * attr
            denom = 1 - exam * car
            self._numerator += num / denom

        self._denominator += 1


class iCCDBNSatClickEM(ParamEM):
    """
    The satisfactoriness parameter of the iCCDBN model without purchase.
    The value of the parameter is inferred using the EM algorithm.
    """

    def update(self, search_session, rank, session_params):
        if search_session.web_results[rank].click and not search_session.web_results[rank].purchase:  # S(uq)' : {cu = 1, tu = 0}
            if rank == search_session.get_last_click_rank():  # (1 - c>r)
                sat_click = session_params[rank][iCCDBN.param_names.sat_click].value()
                cont = session_params[rank][iCCDBN.param_names.cont].value()
                car = session_params[rank + 1][iCCDBN.param_names.car].value() \
                    if rank < len(search_session.web_results) - 1 \
                    else 0

                self._numerator += sat_click / (1 - (1 - sat_click) * cont * car)

            self._denominator += 1


class iCCDBNSatPurEM(ParamEM):
    """
    The satisfactoriness parameter of the iCCDBN model with purchase.
    The value of the parameter is inferred using the EM algorithm.
    """

    def update(self, search_session, rank, session_params):  # TO-DO: KONTROL
        if search_session.web_results[rank].click and search_session.web_results[rank].purchase:  # S(uq)' : {cu = 1, tu = 1}
            if rank == search_session.get_last_click_rank():  # (1 - c>r)
                sat_pur = session_params[rank][iCCDBN.param_names.sat_pur].value()
                cont = session_params[rank][iCCDBN.param_names.cont].value()
                car = session_params[rank + 1][iCCDBN.param_names.car].value() \
                    if rank < len(search_session.web_results) - 1 \
                    else 0

                self._numerator += sat_pur / (1 - (1 - sat_pur) * cont * car)

            self._denominator += 1


class iCCDBNPurMLE(ParamMLE):
    """
    The purchase parameter of the iCCDBN model.
    The value of the parameter is inferred using the MLE algorithm.

    """

    def update(self, search_session, rank, session_params):
        if search_session.web_results[rank].click:  # cu
            if search_session.web_results[rank].purchase:  # tu
                self._numerator += 1
            self._denominator += 1


class iCCDBNContEM(ParamEM):
    """
    The continuation (persistence) parameter of the iCCDBN model.
    The value of the parameter is inferred using the EM algorithm.
    """
    def __init__(self, static):
        self.static = static
        if static:
            self._numerator = static*100
            self._denominator = 100
        else:
            self._numerator = 1
            self._denominator = 2

    def update(self, search_session, rank, session_params):
        if self.static:
            return
        factor = iCCDBN._get_continuation_factor(search_session, rank, session_params)
        total_factor = iCCDBN._get_total_continuation_factor(search_session, rank, session_params)

        def exam_prob(z):
            return factor(1, 0, z) / total_factor  # ESS(z)
        ess_0 = exam_prob(0)
        ess_1 = exam_prob(1)
        self._numerator += ess_1
        self._denominator += ess_0 + ess_1

    def __deepcopy__(self, memo):
        copied = type(self)(getattr(self, 'static', True))
        copied._numerator = getattr(self, '_numerator', 95)
        copied._denominator = getattr(self, '_denominator', 100)
        return copied
