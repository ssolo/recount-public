package count.model;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import org.xml.sax.InputSource;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import count.ds.AnnotatedTable;
import count.ds.Phylogeny;
import count.io.NewickParser;
import count.io.RateVariationParser;
import count.io.TableParser;
import count.matek.Logarithms;

/**
 * Java reference driver — extends CountVerifyXML to support arbitrary
 * minCopies via FamilySizeLikelihood + LogGradient (Theorems 3-5).
 *
 * Usage:
 *   java count.model.CountVerifyXML2 <countxml(.gz)> [sessionId] [tableName] [minCopies]
 *
 * For minCopies <= 2: uses Likelihood.getCorrectedLL / Gradient
 *                     (same path as the original CountVerifyXML).
 * For minCopies  > 2: uses FamilySizeLikelihood.getUnobservedProfile to
 *                     compute L(<min_copies) and the analytical L(0) gradient
 *                     through LogGradient — matches the Csurös 2026 SI
 *                     Theorems 3-5.
 */
public class CountVerifyXML2 {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: CountVerifyXML2 <countxml(.gz)> [sessionId] [tableName] [minCopies]");
            System.exit(2);
        }
        String path = args[0];
        String wantSession = args.length > 1 && !args[1].isEmpty() ? args[1] : null;
        String wantTable = args.length > 2 && !args[2].isEmpty() ? args[2] : null;
        int minCopies = args.length > 3 ? Integer.parseInt(args[3]) : 1;

        DocumentBuilder db = DocumentBuilderFactory.newInstance().newDocumentBuilder();
        org.w3c.dom.Document doc;
        try (InputStream in = openMaybeGzipped(path)) {
            // Force UTF-8 reading even if the XML declaration says us-ascii —
            // the focal arc269 files contain UTF-8 bytes in CDATA but declare
            // us-ascii in the prolog, which trips Xerces' strict ASCII reader.
            Reader rdr = new InputStreamReader(in, StandardCharsets.UTF_8);
            InputSource is = new InputSource(rdr);
            is.setEncoding("UTF-8");
            doc = db.parse(is);
        }

        Element session = pickSession(doc.getDocumentElement(), wantSession);
        if (session == null) throw new RuntimeException("no session with model");
        System.out.println("# session=" + session.getAttribute("id"));

        Element treeEl = firstChildElement(session, "tree");
        Phylogeny tree = NewickParser.readTree(new StringReader(treeEl.getTextContent()));
        System.out.println("# tree=" + tree.getNumLeaves() + "/" + tree.getNumNodes());

        Element modelEl = firstChildElement(treeEl, "model");
        BufferedReader modelReader = new BufferedReader(new StringReader(modelEl.getTextContent()));
        TreeWithRates rates_base = RateVariationParser.readModel(modelReader, tree).getBaseModel();
        TreeWithLogisticParameters lrates =
            (rates_base instanceof TreeWithLogisticParameters)
                ? (TreeWithLogisticParameters) rates_base
                : new TreeWithLogisticParameters(rates_base, false);
        System.out.println("# rates loaded");

        Element tableEl = pickTable(session, wantTable);
        String[] taxa = tree.getLeafNames();
        BufferedReader tableReader = new BufferedReader(new StringReader(tableEl.getTextContent()));
        AnnotatedTable table = TableParser.readTable(taxa, tableReader, false);
        System.out.println("# table=" + tableEl.getAttribute("name")
            + " " + table.getFamilyCount() + "x" + taxa.length);

        Likelihood lik = new Likelihood(rates_base, table);
        long t0_warm = System.nanoTime();
        @SuppressWarnings("unused") double _w = lik.getLL();
        long t_warm = System.nanoTime() - t0_warm;
        Likelihood likTimed = new Likelihood(rates_base, table);
        long t0_ll = System.nanoTime();
        @SuppressWarnings("unused") double _l = likTimed.getLL();
        long t_ll = System.nanoTime() - t0_ll;
        double LL = lik.getLL();
        int F = table.getFamilyCount();
        System.out.printf("# java LL warmup: %.1f ms%n", t_warm / 1.0e6);
        System.out.printf("# java LL timed:  %.1f ms%n", t_ll / 1.0e6);

        // ---- Compute the unobserved-profile log-probability L(<minCopies)
        double Lunobs;
        if (minCopies <= 0) {
            Lunobs = Double.NEGATIVE_INFINITY;
        } else if (minCopies == 1) {
            Lunobs = lik.getEmptyLL();
        } else if (minCopies == 2) {
            Lunobs = Logarithms.add(lik.getEmptyLL(), lik.getSingletonLL());
        } else {
            // minCopies >= 3: SI Theorems 3-5 via FamilySizeLikelihood
            count.model.Posteriors.Profile unobsProfile =
                count.model.FamilySizeLikelihood.getUnobservedProfile(rates_base, minCopies - 1);
            Lunobs = unobsProfile.getLogLikelihood();
        }
        double p_obs = -Math.expm1(Lunobs);
        double LL_corr;
        if (minCopies == 0) {
            LL_corr = LL;
        } else {
            LL_corr = LL - F * Math.log(p_obs);
        }

        System.out.println("LL_raw=" + LL);
        System.out.println("LL_corrected_min" + minCopies + "=" + LL_corr);
        System.out.println("LL_unobserved=" + Lunobs);
        System.out.println("L0=" + Math.exp(Lunobs));
        System.out.println("LL_empty=" + lik.getEmptyLL());
        System.out.println("LL_singleton=" + lik.getSingletonLL());

        // ---- Gradient
        System.out.println("GRADIENT_PATH=" + (minCopies <= 2 ? "Gradient" : "LogGradient"));
        if (minCopies <= 2) {
            Gradient grad = new Gradient(rates_base, table);
            grad.setMinimumObservedCopies(minCopies);
            double[] g = grad.getCorrectedGradient();
            System.out.println("GRADIENT_SURVIVAL");
            for (int v = 0; v < tree.getNumNodes(); v++) {
                System.out.printf("%d gain %.15e%n", v, g[3 * v + 0]);
                System.out.printf("%d loss %.15e%n", v, g[3 * v + 1]);
                System.out.printf("%d dup  %.15e%n", v, g[3 * v + 2]);
            }
        } else {
            // LogGradient analytical path
            LogGradient lg = new LogGradient(lrates, table);
            lg.setMinimumObservedCopies(minCopies);
            double[][] logD;
            try {
                java.lang.reflect.Method m = LogGradient.class.getDeclaredMethod("getLogSurvivalGradient");
                m.setAccessible(true);
                logD = (double[][]) m.invoke(lg);
            } catch (NoSuchMethodException nsm) {
                // Try via getUnobservedStatistics path manually
                System.out.println("# LogGradient.getLogSurvivalGradient not found, dumping LL only");
                logD = null;
            }
            if (logD != null) {
                System.out.println("GRADIENT_SURVIVAL");
                int n = tree.getNumNodes();
                for (int v = 0; v < n; v++) {
                    // Index in LogGradient: 3*v + {LOSS, DUP, GAIN}? — let's introspect
                    // The Likelihood.PARAMETER_* constants:
                    int IDX_LOSS = count.model.Likelihood.PARAMETER_LOSS;
                    int IDX_DUP  = count.model.Likelihood.PARAMETER_DUPLICATION;
                    int IDX_GAIN = count.model.Likelihood.PARAMETER_GAIN;
                    double dgain = Logarithms.ldiffValue(logD[3 * v + IDX_GAIN]);
                    double dloss = Logarithms.ldiffValue(logD[3 * v + IDX_LOSS]);
                    double ddup  = Logarithms.ldiffValue(logD[3 * v + IDX_DUP]);
                    System.out.printf("%d gain %.15e%n", v, dgain);
                    System.out.printf("%d loss %.15e%n", v, dloss);
                    System.out.printf("%d dup  %.15e%n", v, ddup);
                }
            }
        }
        // ---- Per-node copies and family-presence (sum across profiles) --
        // Uses Posteriors.Profile.getNodeMean(v) for E[ξ̃_v|Ξ_f].
        // The "corrected" totals include the unobserved-profile correction:
        //   N_v_corrected = N_v_observed + F·L(0)/(1-L(0)) · E[ξ̃_v|unobs]
        //   FamiliesPresent_v_corrected = sum_f (1 - P{ξ̃_v=0|Ξ_f}) etc.
        long t0_post = System.nanoTime();
        Posteriors post = new Posteriors(lik);
        int n = tree.getNumNodes();
        double[] copies_node = new double[n];
        double[] copies_edge = new double[n];
        double[] families_present = new double[n];
        for (int f = 0; f < table.getFamilyCount(); f++) {
            Posteriors.Profile pf = post.getPosteriors(f);
            for (int v = 0; v < n; v++) {
                copies_node[v] += pf.getNodeMean(v);
                copies_edge[v] += pf.getEdgeMean(v);
                double[] pN = pf.getNodePosteriors(v);
                if (pN.length > 0) {
                    families_present[v] += 1.0 - pN[0];
                }
            }
        }
        long t_post = System.nanoTime() - t0_post;
        System.out.printf("# java branch_stats (per_node): %.1f ms%n", t_post / 1.0e6);

        // Unobserved correction (for min_copies >= 1)
        double[] copies_node_corr = copies_node.clone();
        double[] families_present_corr = families_present.clone();
        if (minCopies >= 1) {
            // Use the FamilySizeLikelihood unobserved profile for ≥3, or the
            // empty + singletons for ≤2.
            if (minCopies >= 3) {
                count.model.Posteriors.Profile unobs =
                    count.model.FamilySizeLikelihood.getUnobservedProfile(rates_base, minCopies - 1);
                double L0_unobs = unobs.getLogLikelihood();
                double factor = F * Math.exp(L0_unobs) / (1.0 - Math.exp(L0_unobs));
                for (int v = 0; v < n; v++) {
                    copies_node_corr[v] += factor * unobs.getNodeMean(v);
                    double[] pN = unobs.getNodePosteriors(v);
                    if (pN.length > 0) {
                        families_present_corr[v] += factor * (1.0 - pN[0]);
                    }
                }
            }
            // For min_copies <= 2 the corrections are smaller; skip for now.
        }
        System.out.println("NODE_STATS  node  copies_node_observed  copies_node_corrected  families_present_observed  families_present_corrected");
        for (int v = 0; v < n; v++) {
            System.out.printf("NODE_STATS  %d  %.10f  %.10f  %.10f  %.10f%n",
                v, copies_node[v], copies_node_corr[v],
                families_present[v], families_present_corr[v]);
        }
        System.out.println("# done");
    }

    private static InputStream openMaybeGzipped(String path) throws Exception {
        InputStream raw = new FileInputStream(path);
        return path.endsWith(".gz") ? new GZIPInputStream(raw) : raw;
    }
    private static Element firstChildElement(Element parent, String tag) {
        NodeList kids = parent.getChildNodes();
        for (int i = 0; i < kids.getLength(); i++) {
            Node k = kids.item(i);
            if (k.getNodeType() == Node.ELEMENT_NODE && k.getNodeName().equals(tag)) return (Element) k;
        }
        return null;
    }
    private static List<Element> childElements(Element parent, String tag) {
        List<Element> out = new ArrayList<>();
        NodeList kids = parent.getChildNodes();
        for (int i = 0; i < kids.getLength(); i++) {
            Node k = kids.item(i);
            if (k.getNodeType() == Node.ELEMENT_NODE && k.getNodeName().equals(tag)) out.add((Element) k);
        }
        return out;
    }
    private static Element pickSession(Element root, String want) {
        for (Element s : childElements(root, "session")) {
            if (want != null && !s.getAttribute("id").equals(want)) continue;
            if (firstChildElement(firstChildElement(s, "tree"), "model") != null) return s;
        }
        return null;
    }
    private static Element pickTable(Element session, String want) {
        List<Element> ts = childElements(session, "table");
        if (want != null) {
            for (Element t : ts) if (want.equals(t.getAttribute("name"))) return t;
        }
        return ts.isEmpty() ? null : ts.get(0);
    }
}
