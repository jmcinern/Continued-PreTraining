#!/usr/bin/env python3
"""
Test file for add_Qwen_separator.py

This test verifies that the add_separator function correctly adds <|endoftext|>
separators at sentence boundaries in Irish text.
"""

import unittest
import sys
import os

# Add the current directory to the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from add_Qwen_separator import add_separator, SEP, add_separator_simple


class TestAddQwenSeparator(unittest.TestCase):
    
    def setUp(self):
        """Set up test data - expected output with separators and input without separators"""
        
        # Expected output with separators (provided by user)
        self.expected_output_with_separators = """d. Soir ó dheas: 0.8 km soir ó dheas de chrosbhealach 78th St SE agus 99th Ave SE.<|endoftext|>
Agus breithniú á dhéanamh ag an Stát iarrtha, chun críocha phointe (f) de mhír 5, féachaint ar comhlíonadh an ceart íosta chun cosanta, cuirfidh an Stát sin san áireamh an fíoras go ndearna an duine lena mbaineann iarracht an ceartas a imghabháil, nó an fíoras gur chinn an duine sin, nuair a bhí an deis aige leigheas a iarraidh i leith an chinnidh a rinneadh in absentia, gan an leigheas sin a iarraidh.<|endoftext|>
coscánú).<|endoftext|>
Ag idirnasc lena mbeidh tíortha i gceist a bhfuil baint acu le níos mó ná réigiún amháin, féadfaidh an modh um bainistiú ar phlódú a chuirfear i bhfeidhm a bheith éagsúil d'fhonn go mbeadh na modhanna a chuirfear i bhfeidhm i gcomhréir leis na modhanna a chuirtear i bhfeidhm sna réigiúin eile lena mbaineann na tíortha sin.<|endoftext|>
Tá sé beartaithe leis an ítim seo na leithreasuithe a fháil a eascraíonn as na táillí maoirseachta a íocann ardáin an-mhór ar líne agus innill chuardaigh an-mhór agus atá riachtanach chun na costais arna dtabhú ag an gCoimisiún i ndáil lena chúraimí maoirseachta i gcomhréir le Rialachán (AE) 2022/2065 .<|endoftext|>
4.6.5 Bonnchúrsa Comhpháirtíochta Mar chuid d'iarrachtaí an Choláiste freastal ar an gcuóta 15% de mhic léinn neamhthraidisiúnta a bheith ar chláir fochéime, rinneadh socrú le coláistí Choiste Gairmoideachais Chathair Bhaile Átha Cliath (CDVEC) chun forbairt agus comhsholáthar a dhéanamh ar Chúrsa Ullmhúcháin Choláiste Saorealaíon le go mbeadh rochtain ann ar raon cúrsaí fochéime i gColáiste na Tríonóide Baile Átha Cliath.<|endoftext|>
Arna dhéanamh sa Bhruiséil, 19 Iúil 2021.<|endoftext|>
Freagrachtaí agus cúraimí an Bhoird um Bonneagar Margaidh Tograí le haghaidh cinnidh ón gComhairle Rialaithe maidir le seirbhísí agus tionscadail bhonneagair an Eurochórais a ullmhú Gan dochar d'fhreagracht an Bhoird Feidhmiúcháin cruinnithe na Comhairle Rialaithe a ullmhú agus do ghnó reatha BCE, ullmhaíonn BBM tograí don Chomhairle Rialaithe chun cinneadh a dhéanamh sna hábhair seo a leanas, a mhéid atá an Chomhairle Rialaithe tar éis tionscadal/bonneagar sonrach a chur de chúram ar BBM, agus sainorduithe choistí CEBC arna mbunú faoi Airteagal 9 de Rialacha Nós Imeachta an Bhainc Cheannais Eorpaigh á gcur san áireamh go hiomlán: an straitéis fhoriomlán, lena n-áirítear raon feidhme na seirbhísí agus tuairiscí seirbhíse a shainiú; saincheisteanna maidir le rialachas tionscadal; cúrsaí airgeadais, lena n-áirítear: príomhghnéithe an chórais airgeadais a mhionsaothrú (go háirithe an buiséad, an méid, an tréimhse a chumhdaíonn an tréimhse ama, maoiniú) i gcomhréir le rialacha an Choiste um Rialú; anailís rialta ar na rioscaí airgeadais a bhfuil an tEurochóras neamhchosanta orthu; na rialacha bainistíochta le haghaidh cuntas arna sealbhú i leabhair BCE agus arna mbainistiú ag BBM thar ceann an Eurochórais; modheolaíocht an chostais; beartas praghsála; agus anailís ar an gcóras dliteanais; an phleanáil fhoriomlán; an creat dlíthiúil leis na bainc cheannais náisiúnta (BCNanna) a sholáthraíonn seirbhísí bonneagair margaidh don Eurochóras nó a chuireann tionscadail bhonneagair an Eurochórais i gcrích leis an Eurochóras (na 'BCNanna a sholáthar'); an creat dlíthiúil le custaiméirí, chomh maith le haon socrú conarthach nó coinníollacha atá le síniú idir an Eurochóras agus páirtithe leasmhara seachtracha; creat um bainistiú riosca; comhaontuithe leibhéil seirbhíse le páirtithe ábhartha; údarú agus tosaíocht a thabhairt d'iarrataí ar athruithe agus straitéisí tástála/imirce; straitéisí nascachta gréasáin; straitéisí bainistithe géarchéime; straitéis agus creataí cibear-athléimneachta agus slándála faisnéise; dliteanas agus éilimh eile; agus go gcomhlíonann rannpháirtithe i seirbhísí bonneagair an Eurochórais na critéir incháilitheachta is infheidhme.<|endoftext|>
an tréimhse tar éis AMA a roghnú sa cheant ach roimh thús na tréimhse seachadta), tá na soláthraithe acmhainneachta roghnaithe faoi réir tacar ceanglas chun a áirithiú go mbeidh a n-acmhainneacht chonraithe ar fáil i dtús na tréimhse seachadta agus ina rannchuidiú le slándáil an tsoláthair.<|endoftext|>
Áirítear i Rialachán Tarmligthe (AE) 2020/2014 díolúine de minimis maidir le sóil choiteanna a ghabhtar le traimlí agus le heangacha geolbhaigh i roinn 3a agus i bhfolimistéar 4 ICES.<|endoftext|>"""        # Test input without separators (remove all <|endoftext|> from expected output)
        self.test_input_without_separators = self.expected_output_with_separators.replace("<|endoftext|>", "")
    def test_add_separator_exact_output(self):
        """Test that add_separator function produces exactly the expected output"""
        
        result = add_separator_simple(self.test_input_without_separators, SEP)
        result_clean = result.replace('\n', '')
        expected_clean = self.expected_output_with_separators.replace('\n', '')
        print(result)
        print(result_clean)
        print(expected_clean)
        self.assertEqual(result_clean, expected_clean)
    
    
   
if __name__ == "__main__":
    print("Running tests for add_Qwen_separator.py...")
    print(f"Using separator: '{SEP}'")
    print()
    
    # Run the tests
    unittest.main(verbosity=2)
